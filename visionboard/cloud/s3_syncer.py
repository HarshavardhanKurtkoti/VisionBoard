import os
import sys
from typing import Optional, List
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
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize S3 client
        Args:
            bucket_name: Optional bucket name, defaults to S3_BUCKET_NAME constant / env
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME", S3_BUCKET_NAME)
        self.s3_client = None
        self._is_available = False
        
        try:
            import boto3
            from botocore.exceptions import NoCredentialsError, ClientError
            
            self.s3_client = boto3.client('s3')
            self._is_available = True
            logging.info(f"Initialized S3 client with bucket: {self.bucket_name}")
        except Exception as e:
            logging.warning(f"S3 client not initialized (S3 features will be skipped): {str(e)}")
            self._is_available = False

    @property
    def is_available(self) -> bool:
        """Check if S3 client is configured and available"""
        return self._is_available and self.s3_client is not None
    
    def sync_folder_to_s3(
        self,
        folder_path: str,
        s3_prefix: Optional[str] = None,
        exclude: Optional[List[str]] = None
    ) -> bool:
        """
        Sync a local folder to S3
        Args:
            folder_path: Path to local folder
            s3_prefix: Prefix for S3 objects (folder path in bucket)
            exclude: List of file patterns to exclude
        Returns:
            bool: True if synced, False if skipped
        """
        if not self.is_available:
            logging.warning("S3 sync requested but S3 client is unavailable. Skipping upload.")
            return False

        try:
            logging.info(f"Syncing folder {folder_path} to S3 bucket {self.bucket_name}")
            
            if not os.path.exists(folder_path):
                logging.warning(f"Folder not found for S3 sync: {folder_path}")
                return False
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if exclude and any(pattern in file for pattern in exclude):
                        continue
                    
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, folder_path).replace("\\", "/")
                    s3_key = f"{s3_prefix.rstrip('/')}/{relative_path}" if s3_prefix else relative_path
                    
                    logging.info(f"Uploading {local_path} to s3://{self.bucket_name}/{s3_key}")
                    self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            
            logging.info(f"Successfully synced folder {folder_path} to S3")
            return True
            
        except Exception as e:
            logging.error(f"Error syncing folder to S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def sync_folder_from_s3(
        self,
        folder_path: str,
        s3_prefix: Optional[str] = None,
        exclude: Optional[List[str]] = None
    ) -> bool:
        """
        Sync files from S3 to a local folder
        Args:
            folder_path: Path to local folder
            s3_prefix: Prefix for S3 objects (folder path in bucket)
            exclude: List of file patterns to exclude
        Returns:
            bool: True if synced, False if skipped
        """
        if not self.is_available:
            logging.warning("S3 download requested but S3 client is unavailable. Skipping.")
            return False

        try:
            logging.info(f"Syncing from S3 bucket {self.bucket_name} to folder {folder_path}")
            os.makedirs(folder_path, exist_ok=True)
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            prefix = s3_prefix if s3_prefix else ""
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    if exclude and any(pattern in s3_key for pattern in exclude):
                        continue
                    
                    relative_path = s3_key[len(prefix):].lstrip('/')
                    local_path = os.path.join(folder_path, relative_path)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    logging.info(f"Downloading s3://{self.bucket_name}/{s3_key} to {local_path}")
                    self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logging.info(f"Successfully synced from S3 to folder {folder_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error syncing from S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """
        Upload a single file to S3
        Args:
            local_path: Path to local file
            s3_key: S3 object key (path in bucket)
        """
        if not self.is_available:
            logging.warning("S3 upload requested but S3 client is unavailable. Skipping.")
            return False

        try:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"File not found: {local_path}")
            
            self.s3_client.upload_file(local_path, self.bucket_name, s3_key)
            logging.info(f"Successfully uploaded file to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logging.error(f"Error uploading file to S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def download_file(self, s3_key: str, local_path: str) -> bool:
        """
        Download a single file from S3
        Args:
            s3_key: S3 object key (path in bucket)
            local_path: Path to save file locally
        """
        if not self.is_available:
            logging.warning("S3 download requested but S3 client is unavailable. Skipping.")
            return False

        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logging.info(f"Successfully downloaded s3://{self.bucket_name}/{s3_key} to {local_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error downloading file from S3: {str(e)}")
            raise VisionBoardException(e, sys)
    
    def delete_file(self, s3_key: str) -> bool:
        """
        Delete a file from S3
        Args:
            s3_key: S3 object key (path in bucket)
        """
        if not self.is_available:
            return False

        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logging.info(f"Successfully deleted s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting file from S3: {str(e)}")
            raise VisionBoardException(e, sys)