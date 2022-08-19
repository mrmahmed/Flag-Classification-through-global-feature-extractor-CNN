import os
import shutil



target_path=r'C:\Users\DR. WAQAR\Downloads\avivco97-attachments\result'

# path=r'C:\Users\DR. WAQAR\Downloads\avivco97-attachments\archive (2)'


countries=os.listdir(target_path)

for country in countries:
    img=os.path.join(target_path,country,country+'.png')
    for i in range(0,100):
        shutil.copyfile(img, os.path.join(target_path,country,country+str(i)+'.png'))
