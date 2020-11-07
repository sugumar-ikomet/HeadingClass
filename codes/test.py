from bs4 import BeautifulSoup
import json
import shutil

def creating_folder():
    with open('D:\\ML_Cobe\\ikomet\\scoring\\New folder_raw\\JTD_TestData45_HTMLCleanup.html', 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')

    html = soup.prettify('latin-1')
    with open("D:\\ML_Cobe\\ikomet\\scoring\\Tem\\JTD.html", "wb") as file:
        file.write(html)

    shutil.copy("D:\\ML_Cobe\\ikomet\\scoring\\Tem\\JTD.html", "D:\\ML_Cobe\\ikomet\\scoring\\raw_html\\")



#import main as ab
#ab.h1_h6_heading_detection()

def creating_buffer():
    with open('D:\\ML_Cobe\\ikomet\\scoring\\structured_html\\JTD.html', encoding='utf8') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, 'lxml')

    print("Soup: ",soup)
    #return soup

#creating_buffer()
#creating_folder()