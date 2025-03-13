import os
import hashlib
import time
import random
import urllib.parse
from icrawler.builtin import GoogleImageCrawler
from icrawler.downloader import ImageDownloader


def get_filename(file_url, default_ext):
    """
    根據 file_url 產生唯一檔案名稱，利用 MD5 並保留 URL 中的副檔名（若有），否則使用 default_ext。
    """
    parsed = urllib.parse.urlparse(file_url)
    basename = os.path.basename(parsed.path)
    if "." in basename:
        ext = basename.split(".")[-1]
        if ext:
            filename = hashlib.md5(file_url.encode("utf-8")).hexdigest() + "." + ext
        else:
            filename = (
                hashlib.md5(file_url.encode("utf-8")).hexdigest() + "." + default_ext
            )
    else:
        filename = hashlib.md5(file_url.encode("utf-8")).hexdigest() + "." + default_ext
    return filename


class CustomImageDownloader(ImageDownloader):
    def __init__(
        self,
        *args,
        downloaded_links_file="downloaded_links.txt",
        downloaded_hashes_file="downloaded_hashes.txt",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.downloaded_links_file = downloaded_links_file
        self.downloaded_hashes_file = downloaded_hashes_file

        # 若記錄檔不存在，先建立空檔案
        for file in [self.downloaded_links_file, self.downloaded_hashes_file]:
            if not os.path.exists(file):
                with open(file, "w", encoding="utf-8") as f:
                    f.flush()

        # 載入已下載連結
        with open(self.downloaded_links_file, "r", encoding="utf-8") as f:
            self.downloaded_links = set(line.strip() for line in f if line.strip())

        # 載入已下載圖片的 MD5 hash 紀錄 {hash: filepath}
        self.downloaded_hashes = {}
        with open(self.downloaded_hashes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    hash_val, path = parts
                    self.downloaded_hashes[hash_val] = path

    def _calculate_md5(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _rewrite_downloaded_links(self):
        """將內存中的 downloaded_links 重新寫入記錄檔"""
        with open(self.downloaded_links_file, "w", encoding="utf-8") as f:
            for link in self.downloaded_links:
                f.write(link + "\n")
            f.flush()

    def download(self, task, default_ext, timeout=5, max_retry=3, **kwargs):
        file_url = task.get("file_url")
        # 如果此連結已下載過，直接跳過
        if file_url in self.downloaded_links:
            print(f"已下載過連結：{file_url}，跳過下載")
            return None

        # 呼叫父類別的 download() 方法（該方法本身不回傳檔案路徑）
        super().download(task, default_ext, timeout, max_retry, **kwargs)
        time.sleep(0.5)  # 等待檔案寫入磁碟

        # 從 storage 目錄中取得最新修改的檔案，假設為本次下載的圖片
        dir_path = self.storage.root_dir
        files = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]
        if not files:
            print(f"下載失敗或檔案不存在：{file_url}")
            return None
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        file_path = files[0]
        print("原始檔案路徑", file_path)

        # 根據 URL 產生新的唯一檔案名稱，並重新命名檔案
        new_name = get_filename(file_url, default_ext)
        new_file_path = os.path.join(dir_path, new_name)
        if file_path != new_file_path:
            try:
                os.rename(file_path, new_file_path)
                print("重新命名為", new_file_path)
                file_path = new_file_path
            except Exception as e:
                print("重新命名失敗:", e)

        # 更新連結記錄（先加入）
        self.downloaded_links.add(file_url)
        with open(self.downloaded_links_file, "a", encoding="utf-8") as f:
            f.write(file_url + "\n")
            f.flush()

        # 計算圖片 MD5 值，檢查是否重複
        md5_hash = self._calculate_md5(file_path)
        if md5_hash in self.downloaded_hashes:
            print(
                f"發現重複圖片 (hash: {md5_hash})，連結：{file_url} 與 {self.downloaded_hashes[md5_hash]}，跳過下載"
            )
            # 從連結記錄中移除此連結，更新檔案
            if file_url in self.downloaded_links:
                self.downloaded_links.remove(file_url)
                self._rewrite_downloaded_links()
            return None
        else:
            self.downloaded_hashes[md5_hash] = file_path
            with open(self.downloaded_hashes_file, "a", encoding="utf-8") as f:
                f.write(f"{md5_hash},{file_path}\n")
                f.flush()
        return file_path


def get_image_count(save_dir):
    # 計算資料夾中符合 jpg、jpeg、png 副檔名的檔案數量
    return len(
        [
            f
            for f in os.listdir(save_dir)
            if os.path.isfile(os.path.join(save_dir, f))
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )


def fetch_images(query, target_num=20, save_dir="./downloaded_images", max_attempts=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    attempts = 0
    last_count = get_image_count(save_dir)
    while get_image_count(save_dir) < target_num and attempts < max_attempts:
        needed = target_num - get_image_count(save_dir)
        print(
            f"目前已下載 {get_image_count(save_dir)} 張，還需要 {needed} 張 (嘗試 {attempts + 1}/{max_attempts})"
        )
        # 改變查詢字串，加入隨機數，增加多樣性
        new_query = f"{query} {random.randint(0, 10000)}"
        google_crawler = GoogleImageCrawler(
            downloader_cls=CustomImageDownloader,
            storage={"root_dir": save_dir},
            parser_threads=1,
            downloader_threads=1,
        )
        google_crawler.crawl(keyword=new_query, max_num=needed)
        time.sleep(1)
        current_count = get_image_count(save_dir)
        if current_count == last_count:
            attempts += 1
        else:
            attempts = 0
            last_count = current_count
    if get_image_count(save_dir) < target_num:
        print(f"下載結束，但僅達到 {get_image_count(save_dir)} 張圖片")
    else:
        print(f"成功下載 {target_num} 張圖片")


if __name__ == "__main__":
    description = input("請輸入物品描述：")
    fetch_images(description, target_num=30)

