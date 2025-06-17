import os
import sys
import json
import base64
import shutil
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv, find_dotenv
from tqdm.asyncio import tqdm
from PIL import Image
import io
import time
import random
from typing import Optional

# 1) Load environment & append your code path (unchanged)
env_path = find_dotenv()
load_dotenv(env_path)
home_path = os.getenv("HOME_PATH")
sys.path.append(os.path.join(home_path, "ICTC"))

# 2) Import your existing args
from utils.argument import args

# 3) Import and configure OpenAI client
from openai import AsyncOpenAI
client = AsyncOpenAI(api_key=os.getenv("API_KEY"))

# Image cache to avoid re-encoding
image_cache = {}

def load_image_paths_from_folder(folder_path):
    exts = (".jpg", ".jpeg", ".png")
    return [
        os.path.join(folder_path, fn)
        for fn in os.listdir(folder_path)
        if fn.lower().endswith(exts)
    ]

def encode_image_optimized(img_path):
    """Optimized image encoding with caching and size optimization"""
    if img_path in image_cache:
        return image_cache[img_path]
    
    with open(img_path, "rb") as img_f:
        raw = img_f.read()
    
    # Optimize image size if too large (OpenAI has size limits)
    img = Image.open(io.BytesIO(raw))
    
    # Resize if image is too large (saves tokens and processing time)
    max_size = 1024
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        buffer = io.BytesIO()
        format_name = img.format if img.format else 'JPEG'
        img.save(buffer, format=format_name, quality=85, optimize=True)
        raw = buffer.getvalue()
    
    # Get MIME type
    mime = img.get_format_mimetype() if hasattr(img, 'get_format_mimetype') else 'image/jpeg'
    
    # Encode to base64
    b64 = base64.b64encode(raw).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    
    # Cache the result
    image_cache[img_path] = data_uri
    return data_uri

class RateLimiter:
    """Smart rate limiter that adapts to API limits"""
    def __init__(self, initial_delay=0.1):
        self.delay = initial_delay
        self.last_request_time = 0
        self.consecutive_errors = 0
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_request_time
            
            if time_since_last < self.delay:
                wait_time = self.delay - time_since_last
                await asyncio.sleep(wait_time)
            
            self.last_request_time = time.time()
    
    def on_success(self):
        # Gradually reduce delay on success
        self.consecutive_errors = 0
        self.delay = max(0.05, self.delay * 0.95)
    
    def on_rate_limit(self, retry_after: Optional[float] = None):
        # Increase delay on rate limit
        self.consecutive_errors += 1
        if retry_after:
            self.delay = max(self.delay, retry_after / 1000.0)  # Convert ms to seconds
        else:
            self.delay = min(5.0, self.delay * 1.5)  # Cap at 5 seconds

# Global rate limiter instance
rate_limiter = RateLimiter()

async def process_single_image_with_retry(img_path, prompt, semaphore, max_retries=10):
    """Process a single image with intelligent retry logic"""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                # Wait based on rate limiter
                await rate_limiter.wait_if_needed()
                
                # Encode image (runs in thread pool to avoid blocking)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    data_uri = await loop.run_in_executor(executor, encode_image_optimized, img_path)
                
                # Call OpenAI API
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": data_uri}}
                            ]
                        }
                    ],
                    max_tokens=500,  # Reduced to save tokens
                    timeout=30.0
                )
                
                out_text = resp.choices[0].message.content
                rate_limiter.on_success()
                
                return {
                    "text": out_text,
                    "image_file": os.path.basename(img_path),
                    "metadata": {"attempts": attempt + 1}
                }
                
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "rate_limit" in error_str.lower() or "429" in error_str
                
                if is_rate_limit:
                    # Extract retry-after time if available
                    retry_after = None
                    if "Please try again in" in error_str:
                        try:
                            # Extract milliseconds from error message
                            ms_str = error_str.split("Please try again in ")[1].split("ms")[0]
                            retry_after = float(ms_str)
                        except:
                            pass
                    
                    rate_limiter.on_rate_limit(retry_after)
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        base_delay = min(30, 2 ** attempt)
                        jitter = random.uniform(0.1, 0.5)
                        wait_time = base_delay + jitter
                        
                        print(f"Rate limit hit for {os.path.basename(img_path)}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                        continue
                
                # Non-rate-limit error or max retries reached
                if attempt == max_retries - 1:
                    print(f"Failed to process {img_path} after {max_retries} attempts: {e}")
                    return {
                        "text": f"Error after {max_retries} attempts: {str(e)}",
                        "image_file": os.path.basename(img_path),
                        "metadata": {"error": True, "attempts": max_retries}
                    }
                
                # For non-rate-limit errors, wait a bit before retry
                if not is_rate_limit and attempt < max_retries - 1:
                    await asyncio.sleep(1)

# Keep the old function name for compatibility
async def process_single_image(img_path, prompt, semaphore):
    return await process_single_image_with_retry(img_path, prompt, semaphore)

async def eval_model_async(args):
    # Prepare input & output paths
    image_files = load_image_paths_from_folder(args.image_folder)
    answers_file = os.path.expanduser(args.step1_result_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Read your textual prompt
    async with aiofiles.open(os.path.join(args.exp_path, "step1_prompt.txt"), "r") as f:
        qs = await f.read()
    qs = qs.strip()

    # More conservative concurrent requests to avoid rate limits
    # Start with fewer concurrent requests for rate-limited APIs
    semaphore = asyncio.Semaphore(3)  # Reduced from 10 to 3
    
    print(f"Processing {len(image_files)} images with max 3 concurrent requests...")
    
    # Process all images concurrently
    tasks = [process_single_image(img_path, qs, semaphore) for img_path in image_files]
    
    # Use smaller batch sizes to handle rate limits better
    results = []
    async with aiofiles.open(answers_file, "w") as ans_file:
        batch_size = 20  # Reduced from 50 to 20
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = await tqdm.gather(*batch_tasks, desc=f"Batch {i//batch_size + 1}")
            
            # Write results immediately to avoid memory buildup
            for result in batch_results:
                await ans_file.write(json.dumps(result) + "\n")
                await ans_file.flush()
            
            results.extend(batch_results)
            
            # Small delay between batches to be extra safe
            if i + batch_size < len(tasks):
                await asyncio.sleep(1)

    # Copy to .jsonl
    target = os.path.join(args.exp_path, "step1_result.jsonl")
    if not os.path.exists(target):
        shutil.copy(answers_file, target)
    
    successful = sum(1 for r in results if not r["metadata"].get("error", False))
    print(f"Completed: {successful}/{len(results)} images processed successfully")
    
    return results

def eval_model(args):
    """Wrapper to run async function"""
    return asyncio.run(eval_model_async(args))

# Alternative: Threaded version if you prefer threads over async
def eval_model_threaded(args):
    """Thread-based parallel processing alternative"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    image_files = load_image_paths_from_folder(args.image_folder)
    answers_file = os.path.expanduser(args.step1_result_path)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    with open(os.path.join(args.exp_path, "step1_prompt.txt"), "r") as f:
        qs = f.read().strip()

    def process_image_sync(img_path):
        try:
            data_uri = encode_image_optimized(img_path)
            
            # Use synchronous OpenAI client
            import openai
            openai.api_key = os.getenv("API_KEY")
            
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": qs},
                            {"type": "image_url", "image_url": {"url": data_uri}}
                        ]
                    }
                ],
                max_tokens=1000,
                timeout=30.0
            )
            
            return {
                "text": resp.choices[0].message.content,
                "image_file": os.path.basename(img_path),
                "metadata": {}
            }
        except Exception as e:
            return {
                "text": f"Error: {str(e)}",
                "image_file": os.path.basename(img_path),
                "metadata": {"error": True}
            }

    with open(answers_file, "w") as ans_file:
        # Use ThreadPoolExecutor with optimal thread count
        max_workers = min(10, len(image_files))  # Adjust based on rate limits
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {executor.submit(process_image_sync, img_path): img_path 
                             for img_path in image_files}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_path), total=len(image_files)):
                result = future.result()
                ans_file.write(json.dumps(result) + "\n")
                ans_file.flush()

    # Copy to .jsonl
    target = os.path.join(args.exp_path, "step1_result.jsonl")
    if not os.path.exists(target):
        shutil.copy(answers_file, target)

if __name__ == "__main__":
    # Choose your preferred method:
    # Method 1: Async (recommended for I/O bound tasks)
    eval_model(args)
    
    # Method 2: Threaded (alternative)
    # eval_model_threaded(args)