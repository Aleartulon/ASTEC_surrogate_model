#!/usr/bin/env python3
"""
Standalone script to authenticate and download datasets.

This script:
1. Authenticates via password
2. Queries all datasets with status 'valid'
3. Downloads each dataset
4. Reports results
"""
#python dataset_download.py --output-dir ../../../../../../scratch/aalelonghi/ROM_datasets_ale/ASTEC/original_hdf5/ --limit 1000
import json
import requests
import sys
import getpass
import numpy as np
import urllib3
import logging
import argparse
import os
import tempfile
import time

from pathlib import Path
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

# Add this to suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("assas_data_downloader")

class AssasAPIClient:
    """Client for interacting with ASSAS Data Hub API."""
    
    def __init__(self, base_url: str, verify_ssl: bool = True):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (e.g., 'http://localhost:5000')
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.verify = verify_ssl
        self.authenticated = False
        self.auth_token = None
        
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate with the API using username/password.
        
        Args:
            username: Username for authentication
            password: Password for authentication
            
        Returns:
            bool: True if authentication successful
        """
        auth_endpoints = [
            '/auth/basic/login',
        ]
        
        auth_data = {
            "username": username,
            "password": password
        }
        
        for endpoint in auth_endpoints:
            try:
                auth_url = urljoin(self.base_url, endpoint)
                logger.info(f"Attempting authentication at: {auth_url}")

                # First try to submit the form
                response = self.session.post(
                    auth_url, data=auth_data, allow_redirects=True
                )
                
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response headers: {dict(response.headers)}")
                
                # Check for successful authentication indicators
                success_indicators = self._check_authentication_success(response)
                
                if success_indicators['is_authenticated']:
                    logger.info("Authentication successful!")
                    for indicator in success_indicators['indicators']:
                        logger.info(f"  - {indicator}")
                    self.authenticated = True
                    return True
                else:
                    logger.error("Authentication failed")
                    for reason in success_indicators['failure_reasons']:
                        logger.error(f"  - {reason}")

                    # Show response preview for debugging
                    if response.text:
                        logger.debug(f"  Response preview: {response.text[:300]}...")

                    return False
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                continue

        logger.error("All authentication endpoints failed")
        return False

    def _check_authentication_success(self, response) -> Dict[str, Any]:
        """
        Check if authentication was successful based on multiple indicators.
        
        Args:
            response: HTTP response object
            
        Returns:
            Dict with authentication status and indicators
        """
        indicators = []
        failure_reasons = []
        is_authenticated = False
        
        # 1. Check for session cookies
        session_cookies = \
            [cookie for cookie in self.session.cookies \
                if any(name in cookie.name.lower() \
                       for name in ['session', 'auth', 'login'])]
        
        if session_cookies:
            indicators.append(f"Session cookie found: {session_cookies[0].name}")
            is_authenticated = True
        else:
            failure_reasons.append("No session cookie found")
        
        # 2. Check HTTP status codes
        if response.status_code == 200:
            indicators.append("HTTP 200 OK response")
        elif response.status_code in [302, 303, 301]:
            location = response.headers.get('location', '')
            if location and '/login' not in location.lower():
                indicators.append(f"Redirected to: {location}")
                is_authenticated = True
            else:
                failure_reasons.append(f"Redirected back to login: {location}")
        else:
            failure_reasons.append(f"HTTP {response.status_code}")
        
        # 3. Check response content
        content_type = response.headers.get('content-type', '')
        
        if content_type.startswith('application/json'):
            try:
                data = response.json()
                if data.get('success') or data.get('status') == 'success':
                    indicators.append("JSON success response")
                    is_authenticated = True
                elif 'error' in data or 'message' in data:
                    failure_reasons.append(
                        f"JSON error: \
                            {data.get('error', data.get('message', 'Unknown'))}"
                    )
            except json.JSONDecodeError:
                failure_reasons.append("Invalid JSON response")
        
        elif content_type.startswith('text/html'):
            response_text = response.text.lower()
            
            # Check for error messages (these indicate failure)
            error_phrases = [
                'invalid username or password',
                'authentication failed',
                'login failed',
                'incorrect credentials',
                'please provide both username and password',
                'error',
                'failed'
            ]
            
            found_errors = \
                [phrase for phrase in error_phrases if phrase in response_text]
            
            if found_errors:
                failure_reasons.extend(
                    [f"Error message found: '{error}'" for error in found_errors]
                )
                is_authenticated = False
            else:
                # Check for success indicators in HTML
                success_phrases = [
                    'welcome',
                    'dashboard',
                    'home',
                    'logout',
                    'successfully',
                    'assas data hub',  # Likely the main page
                    'datasets',  # Likely the dashboard
                ]
                
                found_success = \
                    [phrase for phrase in success_phrases if phrase in response_text]
                
                if found_success:
                    indicators.extend(
                        [f"Success indicator: '{phrase}'" for phrase in found_success]
                    )
                    # If we have both session cookie AND success indicators,
                    # we're authenticated
                    if session_cookies:
                        is_authenticated = True
                else:
                    failure_reasons.append("No success indicators found in HTML")
        
        # 4. Check URL path (if we ended up on a different page)
        final_url = response.url.lower()
        if '/login' in final_url:
            failure_reasons.append("Still on login page")
        elif final_url != urljoin(self.base_url, '/auth/basic/login').lower():
            indicators.append(f"Navigated to different page: {response.url}")
            if session_cookies:
                is_authenticated = True
        
        # 5. Final decision logic
        # If we have session cookies and no explicit error messages,
        # consider it successful
        if session_cookies and not any(
            'error' in reason.lower() for reason in failure_reasons
        ):
            is_authenticated = True
        
        return {
            'is_authenticated': is_authenticated,
            'indicators': indicators,
            'failure_reasons': failure_reasons,
            'has_session_cookie': bool(session_cookies),
            'response_type': content_type
        }

    def get_datasets(
        self,
        status: str = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> Optional[List[Dict]]:
        """
        Get datasets from the API.
        
        Args:
            status: Filter by status (e.g., 'valid')
            limit: Maximum number of datasets to retrieve
            
        Returns:
            List of dataset dictionaries or None if error
        """
        try:
            if not self.authenticated:
                logger.error("Not authenticated")
                return None
            
            datasets_url = urljoin(self.base_url, '/assas_app/datasets')
            params = {'limit': limit, 'offset': offset}

            if status:
                params['status'] = status

            logger.info(f"Querying datasets: {datasets_url} (status={status})")
            response = self.session.get(datasets_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    datasets = data.get('data', {}).get('datasets', [])
                    logger.info(f"Retrieved {len(datasets)} datasets")
                    return datasets
                else:
                    logger.error(f"Failed to get datasets: {data.get('message')}")
                    return None
            else:
                logger.error(f"Failed to get datasets: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting datasets: {e}")
            return None
    
    def get_variable_data(self, dataset_id: str, variable_name: str) -> Optional[Dict]:
        """
        Get variable data from a dataset.
        
        Args:
            dataset_id: UUID of the dataset
            variable_name: Name of the variable to query
            
        Returns:
            Variable data dictionary or None if error
        """
        try:
            if not self.authenticated:
                logger.error("Not authenticated")
                return None

            var_url = urljoin(
                self.base_url, 
                f'/assas_app/datasets/{dataset_id}/data/{variable_name}'
            )

            response = self.session.get(var_url, params={'include_stats': 'true'})
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('data')
                else:
                    return None
            elif response.status_code == 404:
                return None  # Variable not found
            else:
                logger.error(
                    f"Error getting variable data: HTTP {response.status_code}"
                )
                return None
                
        except Exception as e:
            logger.error(f"Error getting variable data: {e}")
            return None
    
    def check_variable_has_valid_data(self, data: Any) -> bool:
        """
        Check if variable data contains non-NaN values.
        
        Args:
            data: Variable data (can be list, number, etc.)
            
        Returns:
            bool: True if contains valid (non-NaN) data
        """
        try:
            if data is None:
                return False
            
            # Handle different data types
            if isinstance(data, (list, tuple)):
                # Flatten nested lists
                flat_data = self._flatten_list(data)
                if not flat_data:
                    return False
                
                # Check for any non-NaN values
                for value in flat_data:
                    if value is not None:
                        try:
                            float_val = float(value)
                            if not (np.isnan(float_val) or np.isinf(float_val)):
                                return True
                        except (ValueError, TypeError):
                            # Non-numeric data is considered valid
                            return True
                return False
                
            elif isinstance(data, (int, float)):
                return not (np.isnan(float(data)) or np.isinf(float(data)))
                
            elif isinstance(data, str):
                return len(data.strip()) > 0
                
            else:
                return data is not None
                
        except Exception:
            return False


def _flatten_list(self, nested_list: List) -> List:
    """Recursively flatten nested lists."""
    flat = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flat.extend(self._flatten_list(item))
        else:
            flat.append(item)
    return flat

def _load_manifest(manifest_path: Path) -> Dict:
    """Load manifest of downloaded datasets (returns dict)."""
    try:
        if manifest_path.is_file():
            with open(manifest_path, 'r', encoding='utf-8') as fh:
                return json.load(fh)
    except Exception:
        pass
    return {'downloaded': {}}

def _save_manifest_atomic(manifest_path: Path, manifest: Dict):
    """Atomically save manifest to disk."""
    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode='w',
            delete=False,
            dir=str(manifest_path.parent),
            encoding='utf-8'
        ) as tf:
            json.dump(manifest, tf, indent=2, default=str)
            tf.flush()
            os.fsync(tf.fileno())
            tmpname = tf.name
        os.replace(tmpname, str(manifest_path))
    except Exception:
        # best-effort; don't raise to avoid breaking downloads
        try:
            with open(manifest_path, 'w', encoding='utf-8') as fh:
                json.dump(manifest, fh, indent=2, default=str)
        except Exception:
            pass


def setup_logging(loglevel: str):
    """Configure logging for the crawler."""
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {loglevel}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler("assas_data_downloader.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main script execution."""
    parser = argparse.ArgumentParser(
        description="ASSAS Data Downloader Script"
    )
    parser.add_argument(
        "--loglevel",
        default="INFO",
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--max-downloads",
        type=int,
        default=0,
        help="Maximum number of datasets to download in this run (0 = no limit)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip datasets already downloaded according "
            "to the manifest or existing file"
        )
    )
    parser.add_argument(
        "--manifest",
        default="downloads/manifest.json",
        help="Path to manifest file that records downloaded dataset UUIDs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of datasets to request from the API"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset to start listing datasets (for paging)"
    )
    parser.add_argument(
    "--output-dir",
    default="downloads",
    help="Directory where downloaded files will be saved"
)
    args = parser.parse_args()

    setup_logging(args.loglevel)
    logger = logging.getLogger("assas_data_downloader")
    logger.info("Starting ASSAS Data Downloader")
    logger.info("=" * 50)
    
    # Configuration
    logger.info("Prompting for API base URL")
    base_url = \
        input("Enter API base URL (default: https://assas.scc.kit.edu): ").strip()
    if not base_url:
        base_url = "https://assas.scc.kit.edu"

    # Authentication
    logger.info("Prompting for username")
    username = input("Username: ").strip()
    if not username:
        logger.error("Username is required")
        sys.exit(1)
    
    password = getpass.getpass("Password: ")
    if not password:
        logger.error("Password is required")
        sys.exit(1)

    # Initialize client
    client = AssasAPIClient(base_url, verify_ssl=False)
    
    # Authenticate
    if not client.authenticate(username, password):
        logger.error("Authentication failed")
        sys.exit(1)
    
    logger.info("Authentication successful")
    
    # Get valid datasets
    logger.info(
        f"Querying datasets with status 'Valid' "
        f"(limit={args.limit}, offset={args.offset})..."
    )
    datasets = client.get_datasets(status='Valid', limit=args.limit, offset=args.offset)
    
    if not datasets:
        logger.error("No valid datasets found or error occurred")
        sys.exit(1)
    
    logger.info(f"Found {len(datasets)} valid datasets")

    # Prepare downloads directory & manifest
    downloads_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)

    # Initialize results
    results = {
        'total_datasets': len(datasets),
        'error_accessing': 0,
        'downloaded_this_run': 0,
        'details': []
    }
    
    logger.info("Checking and downloading datasets...")
    logger.info("-" * 60)

    max_downloads = args.max_downloads \
        if args.max_downloads and args.max_downloads > 0 else None

    for i, dataset in enumerate(datasets, 1):
        if max_downloads is not None and \
            results['downloaded_this_run'] >= max_downloads:
            logger.info(f"Reached max downloads limit ({max_downloads}). Stopping.")
            break

        dataset_id = dataset.get('uuid')
        dataset_name = dataset.get('name', 'Unknown')
        
        logger.info(f"[{i}/{len(datasets)}] Dataset: {dataset_name} (ID: {dataset_id})")
        
        # Skip if no UUID
        if not dataset_id:
            logger.error("Missing dataset UUID, skipping")
            results['error_accessing'] += 1
            results['details'].append({
                'uuid': None,
                'name': dataset_name,
                'status': 'missing_uuid'
            })
            continue

        # If skip-existing and manifest says it was downloaded, skip
        if args.skip_existing and dataset_id in manifest.get('downloaded', {}):
            logger.info(f"Skipping (already in manifest): {dataset_id}")
            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'skipped_manifest'
            })
            continue

        # Construct download URL
        download_path = f"/assas_app/files/download/{dataset_id}"
        download_url = urljoin(client.base_url, download_path)

        # Ensure downloads dir exists
        try:
            downloads_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create downloads directory: {e}")
            results['error_accessing'] += 1
            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'mkdir_failed',
                'error': str(e)
            })
            continue

        # Try to determine expected filename (same logic as before)
        expected_filename = None
        # Try to derive a safe filename without contacting server
        safe_name = "".join(c if c.isalnum() or c in (' ', '.', '-', '_') \
            else '_' for c in dataset_name).strip()
        expected_filename = f"{safe_name}_{dataset_id}.zip"
        expected_path = downloads_dir / expected_filename

        # If skip-existing and file exists on disk, skip and add to manifest if missing
        if args.skip_existing and expected_path.exists():
            logger.info(f"Skipping (file already exists): {expected_path}")
            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'skipped_file',
                'file': str(expected_path)
            })
            # ensure manifest knows about it
            manifest.setdefault('downloaded', {})[dataset_id] = {
                'name': dataset_name,
                'file': str(expected_path),
                'time': time.time()
            }
            _save_manifest_atomic(manifest_path, manifest)
            continue

        # Attempt download (streamed)
        try:
            logger.info(f"Downloading from: {download_url}")
            resp = client.session.get(download_url, stream=True, timeout=60)
            if resp.status_code != 200:
                logger.error(f"Download failed: HTTP {resp.status_code}")
                results['error_accessing'] += 1
                results['details'].append({
                    'uuid': dataset_id,
                    'name': dataset_name,
                    'status': f"http_{resp.status_code}"
                })
                continue

            # Try to determine filename from headers, fall back to name_uuid.zip
            filename = None
            cd = resp.headers.get('content-disposition', '')
            if 'filename=' in cd:
                try:
                    filename = cd.split('filename=')[-1].strip().strip('";\' ')
                except Exception:
                    filename = None

            if not filename:
                filename = expected_filename

            out_path = downloads_dir / filename

            # If skip_existing and out_path exists, skip
            if args.skip_existing and out_path.exists():
                logger.info(
                    f"Skipping (file already exists after server response): {out_path}"
                )
                manifest.setdefault('downloaded', {})[dataset_id] = {
                    'name': dataset_name,
                    'file': str(out_path),
                    'time': time.time()
                }
                _save_manifest_atomic(manifest_path, manifest)
                results['details'].append({
                    'uuid': dataset_id,
                    'name': dataset_name,
                    'status': 'skipped_file_postresp',
                    'file': str(out_path)
                })
                continue

            # Stream write to file
            with open(out_path, 'wb') as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    fh.write(chunk)

            logger.info(f"Downloaded to: {out_path}")
            results.setdefault('downloaded', 0)
            results['downloaded'] += 1
            results['downloaded_this_run'] += 1

            manifest.setdefault('downloaded', {})[dataset_id] = {
                'name': dataset_name,
                'file': str(out_path),
                'time': time.time()
            }
            _save_manifest_atomic(manifest_path, manifest)

            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'downloaded',
                'file': str(out_path)
            })

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error while downloading: {e}")
            results['error_accessing'] += 1
            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'network_error',
                'error': str(e)
            })
        except Exception as e:
            logger.error(f"Unexpected error while downloading: {e}")
            results['error_accessing'] += 1
            results['details'].append({
                'uuid': dataset_id,
                'name': dataset_name,
                'status': 'error',
                'error': str(e)
            })

    # Summary
    logger.info("-" * 60)
    logger.info("Download summary:")
    logger.info(json.dumps(results, indent=2, default=str))
    
if __name__ == "__main__":
    main()