"""
S3 Service for Video Upload and Storage
Handles presigned URL generation and S3 operations
"""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime, timedelta
from config import settings

logger = logging.getLogger(__name__)


class S3Service:
    """
    S3 service for managing video uploads and storage
    """

    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        region: str = "us-east-1",
        bucket_name: Optional[str] = None,
    ):
        """
        Initialize S3 service

        Args:
            access_key_id: AWS access key ID (defaults to settings)
            secret_access_key: AWS secret access key (defaults to settings)
            region: AWS region (defaults to settings)
            bucket_name: S3 bucket name (defaults to settings)
        """
        self.access_key_id = access_key_id or settings.AWS_ACCESS_KEY_ID
        self.secret_access_key = secret_access_key or settings.AWS_SECRET_ACCESS_KEY
        self.region = region or settings.AWS_REGION
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME

        if not self.access_key_id or not self.secret_access_key:
            logger.warning("AWS credentials not configured. S3 operations will fail.")
        
        if not self.bucket_name:
            logger.warning("S3 bucket name not configured. S3 operations will fail.")

        # Initialize S3 client
        self.s3_client = None
        if self.access_key_id and self.secret_access_key:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.access_key_id,
                    aws_secret_access_key=self.secret_access_key,
                    region_name=self.region,
                    config=Config(signature_version='s3v4')
                )
                logger.info(f"S3 client initialized for region: {self.region}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                raise

    def generate_presigned_upload_url(
        self,
        s3_key: str,
        content_type: str,
        expiration: int = 3600,
        max_size_mb: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Generate a presigned URL for uploading a video to S3

        Args:
            s3_key: S3 key (path) where the file will be stored
            content_type: MIME type of the file (e.g., 'video/mp4')
            expiration: URL expiration time in seconds (default: 1 hour)
            max_size_mb: Maximum file size in MB (optional, for validation)

        Returns:
            Dictionary containing:
                - upload_url: Presigned URL for PUT request
                - s3_key: The S3 key where file should be uploaded
                - expires_at: ISO timestamp when URL expires
        """
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized. Check AWS credentials.")

        if not self.bucket_name:
            raise RuntimeError("S3 bucket name not configured.")

        try:
            # Prepare conditions for presigned POST (if needed) or PUT
            # Using PUT for direct upload
            params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'ContentType': content_type,
            }

            # Generate presigned URL
            presigned_url = self.s3_client.generate_presigned_url(
                'put_object',
                Params=params,
                ExpiresIn=expiration
            )

            expires_at = (datetime.utcnow() + timedelta(seconds=expiration)).isoformat()

            logger.info(f"Generated presigned URL for {s3_key}, expires at {expires_at}")

            return {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "s3_bucket": self.bucket_name,
                "expires_at": expires_at,
                "content_type": content_type,
            }

        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            raise RuntimeError(f"Failed to generate presigned URL: {str(e)}")

    def check_object_exists(self, s3_key: str) -> bool:
        """
        Check if an object exists in S3

        Args:
            s3_key: S3 key to check

        Returns:
            True if object exists, False otherwise
        """
        if not self.s3_client or not self.bucket_name:
            return False

        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            logger.error(f"Error checking object existence: {e}")
            raise

    def get_object_metadata(self, s3_key: str) -> Optional[Dict]:
        """
        Get metadata for an S3 object

        Args:
            s3_key: S3 key

        Returns:
            Dictionary with object metadata or None if not found
        """
        if not self.s3_client or not self.bucket_name:
            return None

        try:
            response = self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return {
                "content_type": response.get("ContentType"),
                "content_length": response.get("ContentLength"),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag"),
            }
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            logger.error(f"Error getting object metadata: {e}")
            raise

    def generate_presigned_download_url(
        self,
        s3_key: str,
        expiration: int = 3600
    ) -> str:
        """
        Generate a presigned URL for downloading a file from S3

        Args:
            s3_key: S3 key of the file
            expiration: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned download URL
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 client not initialized or bucket not configured.")

        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return presigned_url
        except ClientError as e:
            logger.error(f"Failed to generate presigned download URL: {e}")
            raise RuntimeError(f"Failed to generate presigned download URL: {str(e)}")
    
    def initiate_multipart_upload(
        self,
        s3_key: str,
        content_type: str
    ) -> Dict[str, str]:
        """
        Initiate a multipart upload to S3
        
        Args:
            s3_key: S3 key where the file will be stored
            content_type: MIME type of the file
            
        Returns:
            Dictionary with upload_id and other metadata
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 client not initialized or bucket not configured.")
        
        try:
            response = self.s3_client.create_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                ContentType=content_type
            )
            
            upload_id = response['UploadId']
            logger.info(f"Initiated multipart upload for {s3_key}, upload_id: {upload_id}")
            
            return {
                "upload_id": upload_id,
                "s3_key": s3_key,
                "s3_bucket": self.bucket_name
            }
        except ClientError as e:
            logger.error(f"Failed to initiate multipart upload: {e}")
            raise RuntimeError(f"Failed to initiate multipart upload: {str(e)}")
    
    def generate_presigned_url_for_part(
        self,
        s3_key: str,
        upload_id: str,
        part_number: int,
        expiration: int = 3600
    ) -> Dict[str, str]:
        """
        Generate a presigned URL for uploading a single part in multipart upload
        
        Args:
            s3_key: S3 key where the file will be stored
            upload_id: Multipart upload ID from initiate_multipart_upload
            part_number: Part number (1-indexed)
            expiration: URL expiration time in seconds
            
        Returns:
            Dictionary with presigned URL and part number
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 client not initialized or bucket not configured.")
        
        try:
            presigned_url = self.s3_client.generate_presigned_url(
                'upload_part',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': s3_key,
                    'UploadId': upload_id,
                    'PartNumber': part_number
                },
                ExpiresIn=expiration
            )
            
            return {
                "upload_url": presigned_url,
                "part_number": part_number,
                "upload_id": upload_id
            }
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for part {part_number}: {e}")
            raise RuntimeError(f"Failed to generate presigned URL for part: {str(e)}")
    
    def complete_multipart_upload(
        self,
        s3_key: str,
        upload_id: str,
        parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Complete a multipart upload
        
        Args:
            s3_key: S3 key where the file is being uploaded
            upload_id: Multipart upload ID
            parts: List of part dictionaries with 'PartNumber' and 'ETag'
            
        Returns:
            Dictionary with completion information
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 client not initialized or bucket not configured.")
        
        try:
            response = self.s3_client.complete_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            logger.info(f"Completed multipart upload for {s3_key}")
            
            return {
                "location": response.get('Location'),
                "bucket": response.get('Bucket'),
                "key": response.get('Key'),
                "etag": response.get('ETag')
            }
        except ClientError as e:
            logger.error(f"Failed to complete multipart upload: {e}")
            raise RuntimeError(f"Failed to complete multipart upload: {str(e)}")
    
    def abort_multipart_upload(
        self,
        s3_key: str,
        upload_id: str
    ) -> None:
        """
        Abort a multipart upload
        
        Args:
            s3_key: S3 key where the file was being uploaded
            upload_id: Multipart upload ID
        """
        if not self.s3_client or not self.bucket_name:
            raise RuntimeError("S3 client not initialized or bucket not configured.")
        
        try:
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket_name,
                Key=s3_key,
                UploadId=upload_id
            )
            logger.info(f"Aborted multipart upload for {s3_key}")
        except ClientError as e:
            logger.error(f"Failed to abort multipart upload: {e}")
            raise RuntimeError(f"Failed to abort multipart upload: {str(e)}")


# Global S3 service instance
s3_service = S3Service()

