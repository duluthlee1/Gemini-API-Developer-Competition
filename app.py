import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import segmentation_models_pytorch as smp
from skimage import measure
import google.generativeai as genai
import json
import re

# 配置Google API Key
api_key = 'your api key'  # 替换成你的API Key
genai.configure(api_key=api_key)

# 加载分割模型
def load_model(model_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# 在图像上绘制边界框，并确保标记位置准确
def draw_bbox(image, bbox, label, color="red", font_size=100):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", font_size)  # 选择字体和大小
    draw.rectangle(bbox, outline=color, width=3)
    draw.text((bbox[0], bbox[1]), label, fill=color, font=font)
    return image

# 进行图像分割并二值化
def segment_image(model, image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # 统一尺寸
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output).squeeze().cpu().numpy()

    # 二值化处理
    binary_output = (output > 0.5).astype(np.uint8) * 255
    return binary_output

# 选出线条，并返回原图上的bbox信息
def select_regions(segmented_image, original_size):
    labels = measure.label(segmented_image)
    properties = measure.regionprops(labels)

    single_line_regions = []
    multi_line_regions = []
    scale_y = original_size[1] / 256
    scale_x = original_size[0] / 256

    for prop in properties:
        bbox = prop.bbox
        original_bbox = (
            int(bbox[1] * scale_x),
            int(bbox[0] * scale_y),
            int(bbox[3] * scale_x),
            int(bbox[2] * scale_y)
        )
        major_axis_length = prop.major_axis_length
        minor_axis_length = prop.minor_axis_length
        num_lines = prop.euler_number  # 使用欧拉数来判断线条数，1表示一个独立线条，0或负数表示多个线条

        region_info = {
            'bbox': original_bbox,
            'length': major_axis_length,
            'width': minor_axis_length,
            'mean_intensity': 0,  # 将强度初始化为0，后续计算时会用到
            'num_lines': num_lines
        }

        if num_lines == 1:
            single_line_regions.append(region_info)
        else:
            multi_line_regions.append(region_info)

    return single_line_regions, multi_line_regions


# 从原始图像中计算每个纳米带的强度
def calculate_intensity_from_original_image(original_image, regions):
    original_image = original_image.convert("L")  # 将图像转换为灰度模式
    np_image = np.array(original_image)

    for region in regions:
        bbox = region['bbox']
        cropped_region = np_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        region['mean_intensity'] = np.mean(cropped_region)

    return regions

# 使用Google Generative AI分析线条信息
def analyze_lines_with_gemini(lines_info):
    prompt = f"""
    Please analyze the following nanobelts and provide a detailed, but concise report. The report should be structured, human-readable, and easy to understand. Include clear labeling on the annotated image, identifying the measurements and the best nanobelt.

    Report Structure:
    1. **Introduction**: Briefly explain the objective of the analysis.
    2. **Nanobelt Analysis**:
        - Identify and label each nanobelt as Line 1, Line 2, etc.
        - For each nanobelt, provide:
            - Length in nanometers (nm)
            - Width in nanometers (nm)
            - Mean intensity
    3. **Comparison**:
        - Compare the nanobelts based on the provided metrics.
        - Identify the nanobelt with the smallest width, longest length, and weakest mean intensity.
    4. **Conclusion**:
        - Highlight the best nanobelt based on the above criteria.
        - Provide a brief explanation as to why this nanobelt is the best choice, considering its length, width, and mean intensity.

    Please ensure the report is clear, concise, and avoids unnecessary technical jargon, making it accessible to a general audience.

    Here is the data for the nanobelts:
    {json.dumps(lines_info, indent=4)}

    Provide the report in a clear and logical format.
    """

    # 创建模型实例
    model = genai.GenerativeModel('gemini-pro')
    
    # 调用生成内容的API
    response = model.generate_content(prompt)

    # 返回生成的结果
    return response.text


# 从Google Gemini的分析结果中提取最佳线条的编号
def extract_best_line(gemini_analysis):
    match = re.search(r'Line (\d+) is the best', gemini_analysis, re.IGNORECASE)
    if match:
        return int(match.group(1)) - 1  # 减去1以匹配Python的0索引
    return None

# 定义segment_and_analyze函数
def segment_and_analyze(image, model, pixel_to_micron_ratio):
    original_size = image.size
    
    segmented_image = segment_image(model, image)
    
    single_line_regions, multi_line_regions = select_regions(segmented_image, original_size)
    
    if not single_line_regions:
        return image, Image.fromarray(segmented_image), "No single-line segments found", "No analysis performed", []

    # 在原始图像上计算每个纳米带的强度
    single_line_regions = calculate_intensity_from_original_image(image, single_line_regions)
    multi_line_regions = calculate_intensity_from_original_image(image, multi_line_regions)

    # 优先选择：宽度最细 -> 长度最长 -> 像素值最弱
    best_line = min(single_line_regions, key=lambda x: (x['width'], -x['length'], x['mean_intensity']))

    annotated_image = image.copy()
    for i, region in enumerate(single_line_regions):
        bbox = region['bbox']
        if region == best_line:
            annotated_image = draw_bbox(annotated_image, bbox, f"Line {i+1} (Best)", color="black", font_size=100)
        else:
            annotated_image = draw_bbox(annotated_image, bbox, f"Line {i+1}", color="red", font_size=100)

    # 标记多条线条的区域
    for i, region in enumerate(multi_line_regions):
        bbox = region['bbox']
        annotated_image = draw_bbox(annotated_image, bbox, f"Multi-Line {i+1}", color="blue", font_size=100)

    # 确保lines_info中的所有数值都是标准Python类型
    lines_info = [{
        'length': float(region['length'] * pixel_to_micron_ratio * 1000),
        'width': float(region['width'] * pixel_to_micron_ratio * 1000),
        'mean_intensity': float(region['mean_intensity']),
        'bbox': region['bbox'],
        'num_lines': int(region['num_lines'])
    } for region in single_line_regions]

    # 使用Google Gemini进行分析
    gemini_analysis = analyze_lines_with_gemini(lines_info)

    return annotated_image, Image.fromarray(segmented_image), {
        "message": "Analyzing all segments.",
        "total_lines": len(single_line_regions)
    }, gemini_analysis, lines_info

# 主界面函数，允许用户选择比例尺类型并上传多张图像
def gradio_interface(uploaded_files, scale_type):
    # 根据选择的比例尺类型计算像素到微米的转换比例
    if scale_type == "5 um":
        pixel_to_micron_ratio = 5 / 141  # 5 um 对应 141 像素
    elif scale_type == "10 um":
        pixel_to_micron_ratio = 10 / 141  # 10 um 对应 141 像素
    else:
        raise ValueError("Invalid scale type selected.")
    
    # 加载模型
    model = load_model(r"C:\Users\dulut\Downloads\best_model_0813.pth")  # 替换为你的模型路径
    
    # 处理多张图片
    all_annotated_images = []
    all_segmented_images = []
    all_results = []
    all_analyses = []
    all_descriptions = []

    for index, file_path in enumerate(uploaded_files):
        image = Image.open(file_path)  # 打开图像文件
        annotated_image, segmented_image, result, analysis, lines_info = segment_and_analyze(image, model, pixel_to_micron_ratio)
        
        description = []
        for i, line_info in enumerate(lines_info):
            try:
                length = line_info['length']
                width = line_info['width']
                mean_intensity = line_info['mean_intensity']
                desc = f"Line {i+1}:\nLength: {length:.2f} nm, Width: {width:.2f} nm, Mean Intensity: {mean_intensity:.2f}"
            except ValueError:
                desc = f"Line {i+1}:\nInvalid data."
            description.append(desc)

        # 组合每张图片的结果并分段显示
        analysis_result = "--- Analysis for Image {} ---\n\n{}\n\n{}\n\n".format(
            index + 1, analysis, "\n".join(description))

        # 保存每张图片的结果
        all_annotated_images.append(annotated_image)
        all_segmented_images.append(segmented_image)
        all_results.append(f"Image {index + 1}: Best Line Analysis\n{analysis_result}")
        all_analyses.append(analysis_result)
    
    # 返回所有图片的结果
    return all_annotated_images, all_segmented_images, "\n\n".join(all_results), "\n".join(all_analyses), "\n\n".join(all_analyses)

# Gradio 接口
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Images for Analysis", file_count="multiple", type="filepath"),  # 支持多张图片上传
        gr.Dropdown(["5 um", "10 um"], label="Select Scale Type")  # 选择比例尺类型
    ],
    outputs=[
        gr.Gallery(label="Annotated Images"),  # 显示多个注释图像
        gr.Gallery(label="Segmented Images"),  # 显示多个分割图像
        gr.Textbox(label="Results", lines=5),  # 显示结果
        gr.Textbox(label="Analysis", lines=10),  # 显示分析
        gr.Textbox(label="Line Descriptions", lines=15)  # 显示纳米带描述
    ],
    title="Automated Image-Based Measurement Tool for Nano Devices",
    description="Upload multiple images and select the appropriate scale type (5 um or 10 um) to segment and analyze using Gemini API."
)

iface.launch(share=True)
