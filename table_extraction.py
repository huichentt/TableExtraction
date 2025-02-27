import gc
import cv2
import easyocr
import pytesseract
import base64
from io import BytesIO
import os
import uvicorn
import warnings

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from PIL import Image
from pdf2image import convert_from_path

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

from ssl_tsr import prediction as ssl_pred
from ssl_tsr import markdownformat

# pre-trained table detection model categories
coco_categories = [{'id':1, 'name':'table', 'supercategory':'table'},
                   {'id':2, 'name':'figure', 'supercategory':'figure'},
                   {'id':3, 'name':'tablecaption', 'supercategory':'tablecaption'},
                   {'id':4, 'name':'figurecaption', 'supercategory':'figurecaption'}]

# easyocr initialize
reader = easyocr.Reader(['en'], gpu=False)
warnings.filterwarnings('ignore')

def apply_ocr(cell_coordinates, tableImage):
    """
    apply easyocr to extract table text row by row
    """
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
      row_text = []
      for cell in row["cells"]:
        cropped_table = Image.open(tableImage)
        cell_image = np.array(cropped_table.crop(cell["cell"]))

        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          text = " ".join([x[1] for x in result])
          row_text.append(text)

      if len(row_text) > max_num_columns:
          max_num_columns = len(row_text)

      data[idx] = row_text
    #print("Max number of columns:", max_num_columns)

    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
          row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

def convert_pdf_to_Image(pdf_file):
    """
    convert pdf file to images
    """
    print(f">> Processing PDF: {pdf_file}")
    imageList = []
    images = convert_from_path(pdf_file)
    pdfFileName = pdf_file.split('/')[-1].split('.')[0]
    for i, img in enumerate(images):
        #if i > 0:
        img.save(f'./tempPDFImages/{pdfFileName}-{i+1}.jpg', 'JPEG')
        imageList.append(f'./tempPDFImages/{pdfFileName}-{i+1}.jpg')
    return imageList

def detectron2_cfg():
    """
    detectron2 config
    """
    cfg = get_cfg()
    cfg.merge_from_file('./saved_models/mask_rcnn_X_101_32x8d_FPN_3x.yaml')
    cfg.MODEL.WEIGHTS = './saved_models/sci3000MarkRcnn4classes/model_0039999.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    return cfg

def convert_img_to_text(image):
    text = pytesseract.image_to_string(image)
    text.replace('\r', '').replace('\n', '')
    return text

# extract the last number before ".jpg"
def extract_number_from_string(s):
    return int(s.split('-')[-1].split('.')[0])

# figure table caption ranking
def figure_table_ranking(prediction_results, index):
    figures = []
    figurecaptions = []
    tables = []
    tablecaptions = []

    for label, bbox in prediction_results:
        if label == 'figure':
            figures.append((label, bbox))
        elif label == 'figurecaption':
            figurecaptions.append((label, bbox))
        elif label == 'table':
            tables.append((label, bbox))
        elif label == 'tablecaption':
            tablecaptions.append((label, bbox))

    figures_sorted = sorted(figures, key=lambda x: x[1][1])
    figurecaptions_sorted = sorted(figurecaptions, key=lambda x: x[1][1])
    tables_sorted = sorted(tables, key=lambda x: x[1][1])
    tablecaptions_sorted = sorted(tablecaptions, key=lambda x: x[1][1])

    ranked_results = {}

    for i, (label, bbox) in enumerate(figures_sorted):
        figure_index = index + i
        ranked_results[figure_index] = bbox
    for i, (label, bbox) in enumerate(figurecaptions_sorted):
        figurecaption_index = index + i
        ranked_results[figurecaption_index] = bbox

    index += len(figures_sorted)

    for i, (label, bbox) in enumerate(tables_sorted):
        table_index = index + i
        ranked_results[table_index] = bbox
    for i, (label, bbox) in enumerate(tablecaptions_sorted):
        tablecaption_index = index + i
        ranked_results[tablecaption_index] = bbox

    return ranked_results

def prediction(detectionDirectory, imageList):
    """
    table figure caption detection from give imageList
    """
    dic = {}
    predictor = DefaultPredictor(detectron2_cfg())
    table_index = 1
    fig_index = 1
    page_table = {}
    page_fig = {}
    imageList = sorted(imageList, key=extract_number_from_string)
    for img in imageList:
        table = []
        table_caption = []
        figure = []
        figure_caption = []
        table_image = []
        im = cv2.imread(img)
        imageName = img.split('.')[1].split('/')[-1]
        page = imageName.split('-')[-1]
        print(page)
        instances = predictor(im)['instances']
        if len(instances) > 0:
            for i in range(0, len(instances)):
                pred_classes = coco_categories[instances[i].pred_classes[0].item()]['name']
                bbox = instances[i].pred_boxes.tensor[0].detach().cpu().numpy()
                cropedImage = Image.open(img).crop(bbox)
                savePath = detectionDirectory + imageName + '_' + pred_classes + '_' + str(i) + '.jpg'
                cropedImage.save(savePath)
                if pred_classes == 'table':                    
                    table.append([pred_classes, bbox.tolist()])
                if pred_classes == 'tablecaption':
                    table_caption.append([pred_classes, bbox.tolist()])
                if pred_classes == 'figure':
                    figure.append([pred_classes, bbox.tolist()])
                if pred_classes == 'figurecaption':
                    figure_caption.append([pred_classes, bbox.tolist()])
            table_page = []
            table_caption_page = []
            figure_page = []
            figure_caption_page = []
            if len(table) > 1:
                table = figure_table_ranking(table, table_index)
                for index in table.keys():
                    table_image.append({index: [convert_pil_to_base64_binary(Image.open(img).crop(table[index])), table[index]]})
                    html = ssl_pred(Image.open(img).crop(table[index]).convert('RGB'))
                    table_page.append({index: markdownformat(html)})
                table_caption = figure_table_ranking(table_caption, table_index)
                for index in table_caption:
                    bbox = table_caption[index]
                    table_caption_page.append([convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox])
                    table_caption_page.append({index: [convert_img_to_text(Image.open(img).crop(bbox)), bbox]})
                table_index += len(table)
            if len(table) == 1:
                bbox = table[0][1]
                table_image = [{table_index: [convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox]}]
                html = ssl_pred(Image.open(img).crop(bbox).convert('RGB'))
                table_page = [{table_index: markdownformat(html)}]
                if len(table_caption) == 1:
                    bbox = table_caption[0][1]
                    table_caption_page = [[convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox], {table_index: [convert_img_to_text(Image.open(img).crop(bbox)), bbox]}]
                table_index += 1
            if len(figure) > 1:
                figure = figure_table_ranking(figure, fig_index)
                for index in figure:
                    bbox = figure[index]
                    figure_page.append({index: [convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox]})
                figure_caption = figure_table_ranking(figure_caption, fig_index)
                for index in figure_caption:
                    bbox = figure_caption[index]
                    figure_caption_page.append({index: [convert_img_to_text(Image.open(img).crop(bbox)), bbox]})
                    figure_caption_page.append({index: [convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox]})
                fig_index += len(figure)
            if len(figure) == 1:
                bbox = figure[0][1]
                figure_page = [{fig_index: [convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox]}]
                if len(figure_caption) == 1:
                    bbox = figure_caption[0][1]
                    figure_caption_page = [{fig_index: [convert_img_to_text(Image.open(img).crop(bbox)), bbox]}, {fig_index: [convert_pil_to_base64_binary(Image.open(img).crop(bbox)), bbox]}]
                fig_index += 1
            dic[page] = {'tableImage': table_image,
                'tableDirectory': table_page,
                    'tableCaptionDirectory': table_caption_page,
                    'figureDirectory': figure_page,
                    'figureCaptionDirectory': figure_caption_page}
    return dic

def convert_pil_to_base64_binary(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    b64_string = base64.b64encode(buffer.read())
    return b64_string.decode('utf-8')

def convert_jpg_to_base64_binary(file_path):
    """
    convert jpg image to binary (base64)
    """
    with open(file_path, "rb") as jpg_img_file:
        b64_string = base64.b64encode(jpg_img_file.read())
        converted = b64_string.decode('utf-8')
    return converted

def generate_temp_file_path(target_file_name):
    """
    generate three temp paths
    """
    tempPDF = './tempPDF'
    if not os.path.exists(tempPDF):
        os.makedirs(tempPDF)
    temp_file_path = os.path.join(tempPDF, f"{target_file_name}")
    detectionResults = './detectionResults/'
    if not os.path.exists(detectionResults):
        os.makedirs(detectionResults)
    tempPDFImages = './tempPDFImages'
    if not os.path.exists(tempPDFImages):
        os.makedirs(tempPDFImages)
    return temp_file_path, detectionResults

def save_the_temp_file(file_contents, temp_file_path):
    """
    save pdf file contents to temp file path
    """
    with open(temp_file_path, 'wb') as fp:
        fp.write(file_contents)

def processing_pdf(pdf_file):
    try:
        target_file_name = pdf_file.filename

        pdf_file_contents = pdf_file.file.read()
        temp_pdf_file_path, detectionDirectory = generate_temp_file_path(target_file_name)
        print(type(temp_pdf_file_path))

        save_the_temp_file(pdf_file_contents, temp_pdf_file_path)

        imageList = convert_pdf_to_Image(temp_pdf_file_path)
        detectronResult = prediction(detectionDirectory, imageList)

        api_response_object = {
            "processed_file": target_file_name,
            "extractions": detectronResult
        }

    except Exception as exception_message:
        print(f"  >> Exception: {exception_message}")
        raise HTTPException(status_code=500, detail=str(exception_message))

    finally:
        pdf_file.file.close()
        #delete temp paths
        # if temp_pdf_file_path:
        #     temp_pdf_path = temp_pdf_file_path.replace(target_file_name, '')
        #     if os.path.exists(temp_pdf_path): shutil.rmtree(temp_pdf_path, ignore_errors=True)
        #     if os.path.exists(detectionDirectory): shutil.rmtree(detectionDirectory, ignore_errors=True)
        #     if os.path.exists(tempPDFImages): shutil.rmtree(tempPDFImages, ignore_errors=True)
    gc.collect()
    return api_response_object

app = FastAPI()

tags_metadata = [
    {
        "name": "Table Extraction API",
        "description": "Table Figure Caption Detection and Content Extraction",
    }
]
app = FastAPI(
    title="Table Figure Understanding API",
)

@app.post("/tableExtraction")
def extraction(pdf_file: UploadFile = File(...)):
    return processing_pdf(pdf_file)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
