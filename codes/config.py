import os

# directory to read raw html files from.Please make sure that only html files are present
# in the folder. No duplicate names for files.
raw_dir ='/home/sysadmin/Desktop/MYCOBE/h1_h6_heading/final_files/HeadingClass/raw_html/'
# directory to store corrected HTML files
rewrite_dir = '/home/sysadmin/Desktop/MYCOBE/h1_h6_heading/final_files/HeadingClass/structured_html/'
# directory to save intermediate files
temp_dir = "/home/sysadmin/Desktop/MYCOBE/h1_h6_heading/final_files/HeadingClass/temp/"
# directory where nltk models are kept
nltk_dir = "/home/sysadmin/Desktop/MYCOBE/h1_h6_heading/final_files/HeadingClass/nltk/"
# directory where model objects etc... are kept
model_objects_dir = "/home/sysadmin/Desktop/MYCOBE/h1_h6_heading/final_files/HeadingClass/model_objects/"

remote_path = os.path.join('/var','www','html','CobeFileServer','Prod')

remote_host = '192.168.1.3'
remote_user = 'cobe'
remote_pass = 'ikomet15'
remote_port = '22'

# names of files kept in 'model_objects_dir'
# Do not change unless you changed the name of files.
le_dir = model_objects_dir+'le_dict.pkl'
ohe_tag_dir = model_objects_dir+'ohe_tag.pkl'
ohe_model_dir = model_objects_dir+'ohe_final.pkl'
tk_dir = model_objects_dir+'tokenizer.pkl'
model_dir = model_objects_dir+'model.h5'
