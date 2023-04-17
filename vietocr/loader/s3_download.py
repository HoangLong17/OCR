import io
import logging
import boto3
import base64
from PIL import Image
import pandas as pd
import json
import os
from io import BytesIO
from tqdm import tqdm
import random
import shutil
from multiprocessing import Pool

class ConfigS3():
    S3_USE_SSL = True
    S3_ENDPOINT = ""
    # S3_ENDPOINT = ""
    S3_STORAGE_SYSTEM = ""
    S3_ACCESS_KEY = ""
    S3_SECRET_KEY = ""
    S3_BUCKET_NAME = ""
    SAVE_IMAGE = True
    S3_PATH_IMAGE = ""
    S3_PATH_SAVE_POST_CHECK = ""
    S3_PATH_SAVE_TEST = ""

    HOST = ""
    PORT = 

class S3StorageService():

    def __init__(self):
        s3_session = boto3.Session(...)
        self.s3_client = s3_session.client(service_name=ConfigS3.S3_STORAGE_SYSTEM,
                                           use_ssl=ConfigS3.S3_USE_SSL,
                                           endpoint_url=ConfigS3.S3_ENDPOINT,
                                           aws_access_key_id=ConfigS3.S3_ACCESS_KEY,
                                           aws_secret_access_key=ConfigS3.S3_SECRET_KEY)
        logging.info(" -- Connect to S3 Storage success.")

    def upload_images(self, file_upload_data, file_path_name_ext, bucket_name):
        """
            params:
                file_upload_data: b64
                bucket_name: datalake
                file_path_name_ext: data/test/1.jpg
            return:
                path
        """
        try:
            file_upload_data = base64str_to_bytesioObj(
                file_upload_data)
            self.s3_client.upload_fileobj(
                file_upload_data, bucket_name, file_path_name_ext)
            logging.debug("File upload success to path: {path}".format(
                path=file_path_name_ext))
            return {"path": file_path_name_ext}
        except Exception as e:
            logging.error(
                "File {path} upload error! -- message: {msg}".format(path=file_path_name_ext, msg=e.__str__()))
            return {"path": ""}

    def download_file(self, bucket_name, file_path_name_ext):
        """
            params:
                bucket_name: datalake
                file_path_name_ext: data/test/1.jpg
            return:
                b64
        """
        try:
            file_obj = self.s3_client.get_object(
                Bucket=bucket_name, Key=file_path_name_ext)
        except Exception as e:
            logging.error(
                "File path {path} do not exits!".format(path=file_path_name_ext))
            return {"b64": ""}
        if not file_obj or not file_obj.get("Body") or not file_obj["Body"]:
            return {"b64": ""}
        try:
            file_data = io.BytesIO(file_obj['Body'].read())
            img_b64 = base64.b64encode(file_data.read()).decode("utf-8")
            logging.debug("File path {path} download success!".format(
                path=file_path_name_ext))
            return {"b64": img_b64}
        except Exception as e:
            logging.error(
                "File path {path} download error! -- message: {msg}".format(path=file_path_name_ext, msg=e.__str__()))
            return {"b64": ""}

    def get_list_file(self, bucket_name, prefix_path):
        paginator = self.s3_client.get_paginator('list_objects_v2')
        """
            params:
                bucket_name: datalake
                prefix_path: raw/GTTT/CIC_9/010333494/
            return:
                []
        """
        try:
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix_path)
            list_img = []
            for page in pages:
                for obj in page['Contents']:
                    image_path = obj['Key']
                    list_img.append(image_path)
            return list_img
        except Exception as e:
            logging.error(
                "File path {path} does not exist! -- message: {msg}".format(path=prefix_path, msg=e.__str__()))
            return []

    def upload_file(self, file_upload_data, file_path_name_ext, bucket_name):
        """
            params:
                file_upload_data: str
                bucket_name: datalake
                file_path_name_ext: data/test/1.xml
            return:
                path
        """
        try:
            self.s3_client.upload_fileobj(
                io.BytesIO(file_upload_data.encode('utf-8')), bucket_name, file_path_name_ext)
            logging.debug("File upload success to path: {path}".format(
                path=file_path_name_ext))
            return {"path": file_path_name_ext}
        except Exception as e:
            logging.error(
                "File {path} upload error! -- message: {msg}".format(path=file_path_name_ext, msg=e.__str__()))
            return {"path": ""}


def base64str_to_bytesioObj(base64str):
    base64_img_bytes = base64str
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    return bytesObj

# Convert Base64 to Image
def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff).convert('RGB')

def b64_2_str(data):
    return base64.b64decode(data).decode('utf-8')


if __name__ == "__main__":
    res = []
    s3 = S3StorageService()
    f = open("path.txt", "w")
    for path in s3.get_list_file(bucket_name=ConfigS3.S3_BUCKET_NAME, prefix_path="dataset/OCR/labeling/cmqd_passport/labels")[1:]:
        json_file = s3.download_file(ConfigS3.S3_BUCKET_NAME, path)['b64']
        data = base64.b64decode(json_file)
        data = data.decode('utf8')
        data = json.loads(data)
        
        res.append(data["task"]["data"]["ocr"].replace("s3://datalake/","")+"\n")
        f.write(data["task"]["data"]["ocr"].replace("s3://datalake/","")+"\n")
    f.close()
    print(res)
