import pandas as pd
import bs4 as bs
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import time
import glob

def main():

	#df=read_csv()

	#print(df)
	googleURL="https://www.google.com/search?q="
	linkedinURL="https://www.linkedin.com/search/results/all/?keywords="
	path=".\\data\\"
	all_files = glob.glob(path + "*.csv")
	print(all_files)
	zipcode=32004

	li = []



	try:
		already_scrapped=pd.read_csv(".\\output_data\\list.csv",squeeze=True)
	except:
		already_scrapped=pd.Series(name='asin')	
	try:
		frame=read_xls()
	except:
		for filename in all_files:
			print(filename)
			df = pd.read_csv(filename, delimiter=",")#, on_bad_lines='skip') #index_col=None, header=0, on_bad_lines='skip')
			li.append(df)
		frame = pd.concat(li, axis=0, ignore_index=True)

	print(f"Already scrapped: \n{already_scrapped}")
	print(f"List to scrap \n{frame}")
	for i,row in frame.iterrows():
		print(row['asin'])
		if row['asin'] in already_scrapped.unique():
			print(f"Skip ASIN {row['asin']}")
		else:
			URL=row['url']
			print(URL)
			#URL="https://www.amazon.com/dp/B083Q1GD9S"
			r= requests.get(URL)
			data=r.text

			#driver.ChromeOptions().add_argument("--incognito")
			options = webdriver.ChromeOptions()
			#options.add_experimental_option("prefs", {
			#    #"download.default_directory": r"yahoo_data",
			#    "download.prompt_for_download": False,
			#    "download.directory_upgrade": True#,
			#})
			options.add_experimental_option('excludeSwitches', ['enable-logging'])
			options.add_argument("--incognito")
			driver = webdriver.Chrome(options=options)
			driver.implicitly_wait(10)
			crumb = None
			driver.get(URL)
			time.sleep(3)


			
			change_to_us_location(driver,zipcode)
			
			try:
				#sellerprofile = driver.find_element_by_id('sellerProfileTriggerId')
				sellerprofile = driver.find_element(By.ID,'sellerProfileTriggerId')
			except NoSuchElementException:
				#frame.drop(index=i, axis=0)
				print("Could not find seller profile, probably a product sold by Amazon")
				#DVWebNode-detail-atf-wrapper DVWebNode
			else:
				sellername=sellerprofile.text
				frame.at[i,'sellername']=sellername
				sellerprofileURL=sellerprofile.get_attribute("href")

				time.sleep(3)
				driver.get(sellerprofileURL)



				try:
					sellerreview=driver.find_elements_by_id('feedback-detail-description')
					aboutseller=driver.find_element_by_id('about-seller')
				except RuntimeError:
					print("RuntimeError")
				except IndexError:
					print("IndexError")
				except NoSuchElementException:
					print("Could not find seller profile, probably a product sold by Amazon warehouse")
					driver.close()
					continue
				else:
					sellerreview=driver.find_elements_by_id('seller-feedback-summary')
				finally:
					pass
				sellerreviewtext=sellerreview[0].text
				frame.at[i,'sellerreview']=sellerreviewtext

				aboutsellertext=aboutseller.text
				frame.at[i,'aboutseller']=aboutsellertext
				print(aboutsellertext)

				elements=driver.find_elements_by_class_name('a-list-item')

				business_adress=""
				for element in elements:
					print(element.text)
					if "Business Name:" in element.text:
						business_name=element.text[14:]
					else:
						if not "Business Address:" in element.text:
							business_adress=business_adress + element.text + " " 
				frame.at[i,'Business Name']=business_name
				frame.at[i,'Business Address']=business_adress
				print(frame)
				frame.at[i,'Google']=make_hyperlink(googleURL,business_name)
				frame.at[i,'LinkedIn']=make_hyperlink(linkedinURL,business_name)
				write_xls(frame)
			#businessaddress=driver.find_elements_by_xpath('//*[@id="seller-profile-container"]/div[2]/div/ul/li[2]/span/ul')
			driver.close()
			frame.drop(i)
			print(row['asin'])
			print(already_scrapped)
			print(pd.Series(data=row['asin']))
			already_scrapped=already_scrapped.append(pd.Series(data=row['asin']), ignore_index=True)
			print(already_scrapped)
			already_scrapped.to_csv(".\\output_data\\list.csv", index=False)

def read_xls():
	return pd.read_excel('./output_data/products.xlsx', sheet_name='Sheet1')

def write_xls(df):
	return df.to_excel('./output_data/products.xlsx')

def change_to_us_location(driver,zipcode):
	try:
		locationbutton=driver.find_element(By.ID,'nav-global-location-popover-link')
		locationbutton.click()
		time.sleep(3)
		zipcodefield=driver.find_element(By.ID,'GLUXZipUpdateInput')
		print(zipcodefield.text)
		zipcodefield.send_keys(zipcode)
		time.sleep(1)
		applybutton=driver.find_element(By.ID,'GLUXZipUpdate')
		applybutton.click()
		time.sleep(3)
		#//*[@id="GLUXConfirmClose-announce"]
		continuebutton=driver.find_element(By.ID,'GLUXConfirmClose')#'//*[@id=\"GLUXConfirmClose\"]'#'GLUXConfirmClose-announce'
		#continuebutton.click()
		driver.execute_script("arguments[0].click();", continuebutton)
		time.sleep(3)
	except NoSuchElementException:
		print("Could not find location button, probably it is a movie")

def make_hyperlink(url,company):
    #print("=HYPERLINK(\""+url + company+"\"; \"#\")")
    urlstring=url + company
    return  	urlstring # % (url.format(urlstring)) #% (url.format(value), value)


if __name__ == "__main__":
	main()