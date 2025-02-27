import os
import sys
import boto3
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

from visionboard.exception.exception import VisionBoardException
from visionboard.logging.logger import logging
from visionboard.constant.training_pipeline import S3_BUCKET_NAME

# Load environment variables
load_dotenv()

class S3Sync:
    """
    Class for syncing files and folders with AWS S3
    """
    
    def __init__(self):
        """Initialize S3 client"""
        try:
            self.s3_client = boto3.client('s3')
            self.bucket_name = S3_BUCKET_NAME
            logging.info(f"Initialized S3 client with bucket: {self.bucket_name}")
        except Exception as e:
            logging.error(f"Error initializing S3 client: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def sync_folder_to_s3(
        self,
        folder_path: str,
        s3_prefix: Optional[str] = None,
        exclude: Optional[list] = None
    ) -> None:
        """
        Sync a local folder to S3
        Args:
            folder_path: Path to local folder
            s3_prefix: Prefix for S3 objects (folder path in bucket)
            exclude: List of file patterns to exclude
        """
        try:
            logging.info(f"Syncing folder {folder_path} to S3")
            
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            # Walk through the folder
            for root, _, files in os.walk(folder_path):
                for file in files:
                    # Skip excluded files
                    if exclude and any(pattern in file for pattern in exclude):
                        continue
                    
                    local_path = os.path.join(root, file)
                    # Create S3 key (path in bucket)
                    relative_path = os.path.relpath(local_path, folder_path)
                    s3_key = os.path.join(s3_prefix, relative_path) if s3_prefix else relative_path
                    
                    # Upload file
                    logging.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            logging.info(f"Successfully synced folder {folder_path} to S3")
            
        except Exception as e:
            logging.error(f"Error syncing folder to S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def sync_folder_from_s3(
        self,
        folder_path: str,
        s3_prefix: Optional[str] = None,
        exclude: Optional[list] = None
    ) -> None:
        """
        Sync files from S3 to a local folder
        Args:
            folder_path: Path to local folder
            s3_prefix: Prefix for S3 objects (folder path in bucket)
            exclude: List of file patterns to exclude
        """
        try:
            logging.info(f"Syncing from S3 to folder {folder_path}")
            
            # Create local folder if it doesn't exist
            os.makedirs(folder_path, exist_ok=True)
            
            # List objects in bucket with prefix
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = s3_prefix if s3_prefix else ""
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip excluded files
                    if exclude and any(pattern in s3_key for pattern in exclude):
                        continue
                    
                    # Create local path
                    relative_path = s3_key[len(prefix):].lstrip('/')
                    local_path = os.path.join(folder_path, relative_path)
                    
                    # Create parent directories if they don't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file
                    logging.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
                    self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logging.info(f"Successfully synced from S3 to folder {folder_path}")
            
        except Exception as e:
            logging.error(f"Error syncing from S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def upload_file(self, local_path: str, s3_key: str) -> None:
        """
        Upload a single file to S3
        Args:
            local_path: Path to local file
            s3_key: S3 object key (path in bucket)
        """
        try:
            logging.info(f"Uploading file {local_path} to S3")
            
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"File not found: {local_path}")
            
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logging.info(f"Successfully uploaded file to s3://{self.bucket_name}/{s3_key}")
            
        except Exception as e:
            logging.error(f"Error uploading file to S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def download_file(self, s3_key: str, local_path: str) -> None:
        """
        Download a single file from S3
        Args:
            s3_key: S3 object key (path in bucket)
            local_path: Path to save file locally
        """
        try:
            logging.info(f"Downloading file from S3")
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logging.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            
        except Exception as e:
            logging.error(f"Error downloading file from S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def delete_file(self, s3_key: str) -> None:
        """
        Delete a file from S3
        Args:
            s3_key: S3 object key (path in bucket)
        """
        try:
            logging.info(f"Deleting file from S3: {s3_key}")
            
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            
        except Exception as e:
            logging.error(f"Error deleting file from S3: {str(e)}")
            raise VisionBoardException(e, sys)
            logger.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {file_path}")
        except ClientError as e:
            logger.error(f"Error downloading file from S3: {str(e)}")
            raise VisionBoardException(e, sys)