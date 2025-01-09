import subprocess
import sys
import logging
import threading
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set up logger
logging.basicConfig(filename='scraping.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Function to ensure all required packages are installed
def install_and_import(package, import_name=None, similar_packages=None):
    """Install and import a package, using a different import name if needed, and try similar packages if not found"""
    import_attempts = [
        (package, package),  # Try original name
        (package, import_name if import_name else package),  # Try specified import name
        (f"python-{package}", package),  # Try with python- prefix
        (f"{package}-python", package),  # Try with -python suffix
    ]
    
    if similar_packages:
        import_attempts.extend([(pkg, pkg) for pkg in similar_packages])
    
    for pkg_name, imp_name in import_attempts:
        try:
            # First try importing
            __import__(imp_name)
            logging.info(f'Successfully imported {imp_name}')
            return True
        except ImportError:
            try:
                # Try installing if import fails
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
                __import__(imp_name)
                logging.info(f'Successfully installed and imported {pkg_name} as {imp_name}')
                return True
            except (subprocess.CalledProcessError, ImportError) as e:
                logging.warning(f'Failed to install/import {pkg_name} as {imp_name}: {e}')
                continue
    
    logging.error(f'Failed to install/import {package} after all attempts')
    return False

# Install packages with special import names and similar packages
packages_with_import_names = {
    'beautifulsoup4': ('bs4', ['beautifulsoup']),
    'googlesearch-python': ('googlesearch', ['google']),
    'python-googlesearch': ('googlesearch', ['google']),
    'google': ('googlesearch', ['googlesearch-python']),
}

# Install and import packages with special names and similar packages first 
for package, (import_name, similar_packages) in packages_with_import_names.items():
    if install_and_import(package, import_name, similar_packages):
        break

# Install remaining packages
required_packages = [
    "requests", "pandas", "tkinter", "selenium", 
    "webdriver_manager", "googletrans==4.0.0-rc1", 
    "tensorflow", "nltk", "opencv-python", "undetected-chromedriver", "fake_useragent"  # Added opencv-python
]

for package in required_packages:
    install_and_import(package)

import requests
from bs4 import BeautifulSoup
import pandas as pd
import tkinter as tk
from tkinter import ttk
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import undetected_chromedriver as uc
from fake_useragent import UserAgent
import time
from googlesearch import search
from googletrans import Translator
import tensorflow as tf
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from textblob import TextBlob

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the translator
translator = Translator()

# Function to detect language and translate text if necessary
def detect_and_translate(text, target_lang='en'):
    try:
        detected_lang = detect(text)
        if (detected_lang != target_lang):
            translated_text = translator.translate(text, dest=target_lang).text
            logging.info(f'Translated text from {detected_lang} to {target_lang}: {translated_text}')
            return translated_text
        return text
    except Exception as e:
        logging.error(f'Error detecting/translating language: {e}')
        return text

# Function to clean and preprocess text
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
        preprocessed_text = ' '.join(tokens)
        logging.info(f'Preprocessed text: {preprocessed_text}')
        return preprocessed_text
    except Exception as e:
        logging.error(f'Error preprocessing text: {e}')
        return text

# Function to extract the address from the company's website imprint
def get_address_from_impressum(soup):
    try:
        possible_terms = ['Address', 'Contact', 'Impressum', 'Contact details']
        for term in possible_terms:
            address_element = soup.find(text=lambda text: term in text)
            if address_element:
                next_element = address_element.find_next()
                while next_element and not next_element.name:
                    next_element = next_element.find_next()
                if next_element:
                    address_text = next_element.text.strip()
                    address_text = detect_and_translate(address_text)
                    address_text = preprocess_text(address_text)
                    logging.info(f'Extracted address: {address_text}')
                    return address_text
    except Exception as e:
        logging.error(f'Error extracting address: {e}')
    return None

# Function to perform sentiment analysis on text
def analyze_sentiment(text):
    try:
        text_blob = TextBlob(text)
        sentiment = text_blob.sentiment
        logging.info(f'Analyzed sentiment: {sentiment}')
        return sentiment
    except Exception as e:
        logging.error(f'Error analyzing sentiment: {e}')
        return None

def setup_driver():
    """Set up an undetected Chrome instance with anti-bot detection measures"""
    options = uc.ChromeOptions()
    
    # Add common browser features to appear more human-like
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options.add_argument('--lang=en-US,en')
    
    # Random user agent
    ua = UserAgent()
    options.add_argument(f'--user-agent={ua.random}')
    
    # Add some window size randomization
    width = random.randint(1024, 1920)
    height = random.randint(768, 1080)
    options.add_argument(f'--window-size={width},{height}')
    
    try:
        driver = uc.Chrome(options=options)
        
        # Execute CDP commands to modify browser fingerprints
        driver.execute_cdp_cmd('Network.setUserAgentOverride', {
            "userAgent": ua.random,
            "platform": random.choice(['Windows', 'Macintosh', 'Linux'])
        })
        
        # Add random delay between actions
        driver.implicitly_wait(random.uniform(3.0, 7.0))
        logging.info('Driver setup completed with random user agent and window size.')
        return driver
    except Exception as e:
        logging.error(f"Error setting up driver: {e}")
        return None

def human_like_interaction(driver, element=None):
    """Simulate human-like behavior"""
    # Random scrolling
    for _ in range(random.randint(2, 5)):
        driver.execute_script(f"window.scrollBy(0, {random.randint(100, 500)})")
        time.sleep(random.uniform(0.5, 2.0))
    
    if element:
        # Move mouse to element with random offset
        offset_x = random.randint(-10, 10)
        offset_y = random.randint(-10, 10)
        driver.execute_script(f"""
            var element = arguments[0];
            var rect = element.getBoundingClientRect();
            var x = rect.left + rect.width/2 + {offset_x};
            var y = rect.top + rect.height/2 + {offset_y};
            var mouseMoveEvent = new MouseEvent('mousemove', {{
                clientX: x,
                clientY: y,
                bubbles: true
            }});
            element.dispatchEvent(mouseMoveEvent);
        """, element)
        time.sleep(random.uniform(0.1, 0.5))
        logging.info(f'Moved mouse to element with offset ({offset_x}, {offset_y}).')

# Function to find the company's website with address, postal code, city, and country
def find_company_website(company_name, address, plz, ort, land, retries=3, instructions_file=None):
    """Modified website search function with enhanced NLP capabilities and learning from instructions"""
    if land:
        land_translated = translator.translate(land, dest='en').text
        query = f'{company_name} {address} {plz} {ort} {land_translated} Impressum'
        
        for attempt in range(retries):
            driver = None
            try:
                driver = setup_driver()
                if not driver:
                    continue
                
                # Random delay before search
                time.sleep(random.uniform(2.0, 5.0))
                
                # Use a different search engine randomly
                search_engines = [
                    "https://www.google.com/search?q=",
                    "https://www.bing.com/search?q=",
                    "https://duckduckgo.com/?q="
                ]
                search_url = random.choice(search_engines) + query
                logging.info(f'Searching for company website with query: {query} using {search_url}')
                
                driver.get(search_url)
                
                # Apply instructions if provided
                if instructions_file:
                    steps = read_instructions(instructions_file)
                    if apply_instructions(driver, steps, company_name, {
                        "address": address,
                        "plz": plz,
                        "ort": ort,
                        "land": land
                    }):
                        return f'screenshots/{company_name}.png'
                else:
                    # Simulate human-like behavior
                    human_like_interaction(driver)
                
                # Handle cookie consent if present (modify selectors as needed)
                try:
                    cookie_buttons = WebDriverWait(driver, 5).until(
                        EC.presence_of_all_elements_located((By.XPATH, 
                            "//button[contains(text(), 'Accept') or contains(text(), 'Agree') or contains(text(), 'Cookies')]"
                        ))
                    )
                    if cookie_buttons:
                        human_like_interaction(driver, cookie_buttons[0])
                        cookie_buttons[0].click()
                        logging.info('Clicked on cookie consent button.')
                except:
                    pass

                # Take screenshot
                screenshot_path = f'screenshots/{company_name}.png'
                time.sleep(random.uniform(1.0, 2.0))
                driver.save_screenshot(screenshot_path)
                logging.info(f'Screenshot saved: {screenshot_path}')
                
                return screenshot_path
                
            except Exception as e:
                logging.error(f'Error during search attempt {attempt + 1} for {company_name}: {e}')
                time.sleep(random.uniform(5.0, 10.0))  # Longer delay between retries
            finally:
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                        
        logging.error(f'Failed to retrieve website for {company_name} after {retries} attempts.')
        return None

# Function to archive unsuccessful screenshots instead of deleting them
def archive_unsuccessful_screenshots(unsuccessful_screenshots):
    archive_dir = 'screenshots/archived'
    os.makedirs(archive_dir, exist_ok=True)
    
    for screenshot in unsuccessful_screenshots:
        try:
            if os.path.exists(screenshot):
                archive_path = os.path.join(archive_dir, f'archived_{os.path.basename(screenshot)}')
                os.rename(screenshot, archive_path)
                logging.info(f'Screenshot {screenshot} archived to {archive_path}')
        except Exception as e:
            logging.error(f'Error archiving screenshot {screenshot}: {e}')

class ScraperModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100, 100, 3)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def process_image(self, image_path):
        img = Image.open(image_path).resize((100, 100))
        return np.array(img)

    def train(self, successful_screenshots, unsuccessful_screenshots):
        x_train = np.array([self.process_image(img) for img in successful_screenshots])
        y_train = np.ones(len(successful_screenshots))
        x_test = np.array([self.process_image(img) for img in unsuccessful_screenshots])
        y_test = np.zeros(len(unsuccessful_screenshots))

        self.model.fit(
            np.concatenate([x_train, x_test]),
            np.concatenate([y_train, y_test]),
            epochs=5
        )

        self.model.save('screenshot_classifier.h5')
        logging.info('Model training completed and saved.')

# Example usage in the ScraperGUI class
class ScraperGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Unternehmensdaten-Scraper')
        self.root.geometry("400x300")
        self.is_running = False
        self.model = ScraperModel()
        self.setup_gui()

    def setup_gui(self):
        self.progress = ttk.Progressbar(self.root, orient='horizontal', length=300, mode='determinate')
        self.progress.pack(pady=20)

        self.stop_button = ttk.Button(self.root, text="Emergency Stop", command=self.stop_scraping)
        self.stop_button.pack(pady=10)

        self.status_label = ttk.Label(self.root, text="Ready")
        self.status_label.pack(pady=10)

        # Start scraping automatically
        self.start_scraping()

    def start_scraping(self):
        if not self.is_running:
            self.is_running = True
            threading.Thread(target=self.scrape_data, daemon=True).start()

    def stop_scraping(self):
        self.is_running = False
        self.status_label.config(text="Stopped")

    def scrape_data(self):
        try:
            df = pd.read_excel('unternehmen.xlsx')
            total = len(df)

            successful_screenshots = []
            unsuccessful_screenshots = []

            for index, row in df.iterrows():
                if not self.is_running:
                    break

                if pd.isna(row.get('Firmenname')):
                    continue

                screenshot_path = find_company_website(
                    row.get('Firmenname', ''),
                    row.get('Straße', ''),
                    row.get('PLZ', ''),
                    row.get('Ort', ''),
                    row.get('Land', '')
                )

                if screenshot_path:
                    successful_screenshots.append(screenshot_path)
                    logging.info(f"Screenshot saved: {screenshot_path}")
                else:
                    unsuccessful_screenshots.append(screenshot_path)
                    logging.warning(f"Failed to capture screenshot for {row.get('Firmenname')}")

                self.progress['value'] = (index + 1) / total * 100
                self.root.update_idletasks()

            if successful_screenshots and unsuccessful_screenshots:
                self.model.train(successful_screenshots, unsuccessful_screenshots)

            df.to_excel('unternehmen_aktualisiert.xlsx', index=False)
            messagebox.showinfo("Success", "Scraping completed!")

        except Exception as e:
            logging.error(f"Error during scraping: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_running = False

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        filename='scraping.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )

    # Download NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Create screenshots directory if it doesn't exist
    os.makedirs('screenshots', exist_ok=True)

    # Start application
    app = ScraperGUI()
    app.run()

    # Start scraping automatically
    app.start_scraping()

# Function to train the model
def train_model(successful_screenshots, unsuccessful_screenshots):
    """Train a model to classify screenshots as successful or unsuccessful"""
    import cv2
    import numpy as np
    
    def load_and_preprocess_images(image_paths, target_size=(224, 224)):
        images = []
        for img_path in image_paths:
            try:
                # Read and resize image
                img = cv2.imread(img_path)
                if img is None:
                    logging.error(f"Could not load image: {img_path}")
                    continue
                img = cv2.resize(img, target_size)
                img = img / 255.0  # Normalize pixel values
                images.append(img)
                logging.info(f'Loaded and preprocessed image: {img_path}')
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")
                continue
        return np.array(images)

    # Prepare data
    try:
        x_train = load_and_preprocess_images(successful_screenshots)
        x_test = load_and_preprocess_images(unsuccessful_screenshots)
        
        if len(x_train) == 0 or len(x_test) == 0:
            logging.error("No valid images found for training")
            return
            
        # Create labels (1 for successful, 0 for unsuccessful)
        y_train = np.ones(len(x_train))
        y_test = np.zeros(len(x_test))
        
        # Create the model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),  # RGB images
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        # Train the model
        history = model.fit(
            x_train, y_train,
            epochs=10,  # Increased epochs for better training
            validation_data=(x_test, y_test),
            verbose=1
        )
        
        # Save the model
        model.save('screenshot_classifier.h5')
        logging.info('Model training completed and saved.')
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")

# Example function calls - replace delete with archive
successful_screenshots = ['screenshot1.png', 'screenshot2.png']
unsuccessful_screenshots = ['screenshot3.png', 'screenshot4.png']
archive_unsuccessful_screenshots(unsuccessful_screenshots)
train_model(successful_screenshots, unsuccessful_screenshots)

# Read Excel file
excel_file = 'companies.xlsx'  # Path to your Excel file
df = pd.read_excel(excel_file)

# Set up GUI and progress bar
root = tk.Tk()
root.title('Company Data Scraper')
root.geometry("400x300")

progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
progress.pack(pady=20)

def save_data_instantly():
    """Save data to Excel file with a temporary name to avoid write conflicts"""
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        temp_filename = f'companies_updated_{timestamp}.xlsx'
        df.to_excel(temp_filename, index=False)
        
        # Safely rename the file to the final name
        if os.path.exists('companies_updated.xlsx'):
            os.remove('companies_updated.xlsx')
        os.rename(temp_filename, 'companies_updated.xlsx')
        logging.info('Data saved successfully')
    except Exception as e:
        logging.error(f'Error saving data: {e}')

# Function to update the data
def update_data(index, row, instructions_file=None):
    """Update data for a given company"""
    if 'CompanyName' not in df.columns:
        logging.error('Column CompanyName not found in the DataFrame.')
        return
        
    company_name = row['CompanyName']
    address = row['Street'] if not pd.isna(row['Street']) else ""
    plz = row['PostalCode'] if not pd.isna(row['PostalCode']) else ""
    ort = row['City'] if not pd.isna(row['City']) else ""
    land = row['Country'] if not pd.isna(row['Country']) else ""

    logging.info(f'Starting data update for {company_name}')

    # Use automated search with instructions
    found_address, company_website = find_company_website(
        company_name, address, plz, ort, land, instructions_file=instructions_file
    )
    
    if found_address:
        df.at[index, 'Street'] = found_address
        logging.info(f'Address for {company_name} updated from {company_website}.')
    else:
        logging.info(f'Address for {company_name} not found.')

    # Fill out missing data
    if not address:
        df.at[index, 'Street'] = found_address if found_address else "N/A"
    if not plz:
        df.at[index, 'PostalCode'] = "N/A"
    if not ort:
        df.at[index, 'City'] = "N/A"
    if not land:
        df.at[index, 'Country'] = "N/A"

    # Save changes immediately after each update
    save_data_instantly()
    
    logging.info(f'Completed data update for {company_name}.')

# Function to display progress and collect data
def learn_phase():
    """Simulate a learning phase where the script learns from provided instructions and past data"""
    logging.info('Starting learning phase...')
    
    # Analyze past data to improve the model
    try:
        # Load past data
        past_data_file = 'past_data.xlsx'
        if os.path.exists(past_data_file):
            past_df = pd.read_excel(past_data_file)
            logging.info('Loaded past data for analysis.')
            
            # Analyze past data to find patterns and improve the model
            # For example, identify common issues, successful patterns, etc.
            common_issues = past_df['Error'].value_counts().head(5)
            successful_patterns = past_df[past_df['Status'] == 'Success']['Pattern'].value_counts().head(5)
            logging.info(f'Common issues: {common_issues}')
            logging.info(f'Successful patterns: {successful_patterns}')
            
            # Use insights to improve the model
            # For example, retrain the model with additional data or adjust parameters
            def retrain_model():
                # Implement your model training logic here
                # Example: Retrain the model with additional data or adjust parameters
                logging.info('Retraining the model with new insights...')
                
                # Load and preprocess images
                successful_screenshots = past_df[past_df['Status'] == 'Success']['ScreenshotPath'].tolist()
                unsuccessful_screenshots = past_df[past_df['Status'] == 'Failure']['ScreenshotPath'].tolist()
                x_train = load_and_preprocess_images(successful_screenshots)
                x_test = load_and_preprocess_images(unsuccessful_screenshots)
                
                if len(x_train) == 0 or len(x_test) == 0:
                    logging.error("No valid images found for training")
                    return
                
                # Create labels (1 for successful, 0 for unsuccessful)
                y_train = np.ones(len(x_train))
                y_test = np.zeros(len(x_test))
                
                # Create the model
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(224, 224, 3)),  # RGB images
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                # Compile the model
                model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=['accuracy']
                )
                
                # Train the model
                history = model.fit(
                    x_train, y_train,
                    epochs=5,
                    validation_data=(x_test, y_test),
                    verbose=1
                )
                
                # Save the model
                model.save('screenshot_classifier.h5')
                logging.info('Model retraining completed and saved.')
            
            retrain_model()
            
            logging.info('Analysis of past data completed.')
        else:
            logging.warning('Past data file not found. Skipping analysis.')
        
        # Train or retrain the model based on new insights
        # Implement your model training logic here
        # ...
        
        logging.info('Model training/retraining completed.')
        
    except Exception as e:
        logging.error(f'Error during learning phase: {e}')
    
    # Optimize the scraping process based on new insights
    # For example, adjust scraping strategies, improve error handling, etc.
    def optimize_scraping_process():
        # Implement your optimization logic here
        # Example: Adjust scraping strategies, improve error handling, etc.
        logging.info('Optimizing the scraping process...')
        # ...optimization logic...
        logging.info('Scraping process optimization completed.')
    
    optimize_scraping_process()
    
    logging.info('Learning phase completed.')

def work_phase(index, row, instructions_file=None):
    """Simulate a working phase where the script updates data for a given company"""
    logging.info('Starting working phase...')
    update_data(index, row, instructions_file)
    logging.info('Working phase completed.')

def scrape_data(instructions_file=None):
    """Scrape data for all companies with a 60 to 40 learn-work ratio"""
    if 'CompanyName' not in df.columns:
        logging.error('Column CompanyName not found in the DataFrame.')
        return
    
    threads = []
    total_companies = len(df)
    learn_work_ratio = 0.6  # 60% learning, 40% working
    
    # Create a lock for thread-safe file operations
    file_lock = threading.Lock()
    
    def update_data_thread_safe(index, row):
        with file_lock:
            work_phase(index, row, instructions_file)
    
    for index, row in df.iterrows():
        if pd.isna(row['CompanyName']):
            continue  # Skip if the company name is missing
        
        # Determine if the current phase should be learning or working
        if random.random() < learn_work_ratio:
            learn_phase()
        else:
            thread = threading.Thread(target=update_data_thread_safe, args=(index, row))
            threads.append(thread)
            thread.start()
        
        # Add a small delay between thread starts to avoid overwhelming resources
        time.sleep(0.1)
    
    # Update progress bar
    for i, thread in enumerate(threads):
        thread.join()
        progress['value'] = (i + 1) / total_companies * 100
        root.update_idletasks()

    logging.info('All data processing completed!')

    # Complete progress bar
    progress['value'] = 100
    root.update_idletasks()

import re

def read_instructions(file_path):
    """Read instructions from a text file and return them as a list of steps"""
    try:
        with open(file_path, 'r') as file:
            steps = file.readlines()
        steps = [parse_instruction(step.strip()) for step in steps if step.strip()]
        return steps
    except Exception as e:
        logging.error(f'Error reading instructions from {file_path}: {e}')
        return []

def parse_instruction(instruction):
    """Parse a Q&A formatted instruction into a structured step"""
    instruction = instruction.lower()
    if "open browser" in instruction:
        return "open_browser"
    elif "type in name of the company" in instruction:
        return "type_company_name"
    elif "search name of company" in instruction:
        return "search_company_name"
    elif "open first search link" in instruction:
        return "open_first_link"
    elif "compare company info" in instruction:
        return "compare_info"
    elif "fill out missing data" in instruction:
        return "fill_missing_data"
    elif "try next search link" in instruction:
        return "try_next_link"
    logging.warning(f'Unknown instruction format: {instruction}')
    return None

def apply_instructions(driver, steps, company_name, company_info):
    """Apply the instructions to the driver"""
    try:
        for step in steps:
            if step is None:
                continue
            if step == "open_browser":
                driver.get("about:blank")
            elif step == "type_company_name":
                search_box = driver.find_element(By.NAME, "q")
                search_box.send_keys(company_name)
            elif step == "search_company_name":
                search_box.send_keys(Keys.RETURN)
            elif step == "open_first_link":
                first_link = driver.find_element(By.CSS_SELECTOR, "a")
                first_link.click()
            elif step == "compare_info":
                if compare_company_info(driver, company_info):
                    return True
            elif step == "fill_missing_data":
                fill_missing_data(driver, company_info)
            elif step == "try_next_link":
                next_link = driver.find_element(By.CSS_SELECTOR, "a")
                next_link.click()
    except Exception as e:
        logging.error(f'Error applying instructions: {e}')
    return False

def compare_company_info(driver, company_info):
    """Compare the company info from the website with the given company info"""
    try:
        # Extract text content from the webpage
        page_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        
        # Preprocess the text content
        page_text = preprocess_text(page_text)
        
        # Extract company info
        address = company_info.get('address', '').lower()
        plz = company_info.get('plz', '').lower()
        ort = company_info.get('ort', '').lower()
        land = company_info.get('land', '').lower()
        
        # Check if at least 80% of the information matches
        match_count = 0
        total_fields = 4
        
        if address in page_text:
            match_count += 1
        if plz in page_text:
            match_count += 1
        if ort in page_text:
            match_count += 1
        if land in page_text:
            match_count += 1
        
        match_percentage = (match_count / total_fields) * 100
        logging.info(f'Match percentage: {match_percentage}%')
        
        return match_percentage >= 80
    except Exception as e:
        logging.error(f'Error comparing company info: {e}')
        return False

def fill_missing_data(driver, company_info):
    """Fill out missing data with the information from the website"""
    try:
        # Extract text content from the webpage
        page_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        
        # Preprocess the text content
        page_text = preprocess_text(page_text)
        
        # Extract company info
        address = company_info.get('address', '').lower()
        plz = company_info.get('plz', '').lower()
        ort = company_info.get('ort', '').lower()
        land = company_info.get('land', '').lower()
        
        # Fill missing data
        if not address:
            address_match = re.search(r'\baddress\b:\s*(.*)', page_text)
            if address_match:
                company_info['address'] = address_match.group(1).strip()
                logging.info(f'Filled missing address: {company_info["address"]}')
        
        if not plz:
            plz_match = re.search(r'\bpostal code\b:\s*(\d+)', page_text)
            if plz_match:
                company_info['plz'] = plz_match.group(1).strip()
                logging.info(f'Filled missing postal code: {company_info["plz"]}')
        
        if not ort:
            ort_match = re.search(r'\bcity\b:\s*(.*)', page_text)
            if ort_match:
                company_info['ort'] = ort_match.group(1).strip()
                logging.info(f'Filled missing city: {company_info["ort"]}')
        
        if not land:
            land_match = re.search(r'\bcountry\b:\s*(.*)', page_text)
            if land_match:
                company_info['land'] = land_match.group(1).strip()
                logging.info(f'Filled missing country: {company_info["land"]}')
        
    except Exception as e:
        logging.error(f'Error filling missing data: {e}')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Company Data Scraper')
    parser.add_argument('--instructions', type=str, help='Path to the instructions file')
    args = parser.parse_args()
    
    # Start scraping with instructions if provided
    scraping_thread = threading.Thread(target=scrape_data, args=(args.instructions,))
    scraping_thread.start()
    
    # Start GUI
    root.mainloop()