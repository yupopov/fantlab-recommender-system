import re
import os

def get_work_ids(data_dir='data/raw', out_file='work_ids.txt'):
    
    html_files = [os.path.join(data_dir, file_) for file_ in os.listdir(data_dir) if file_.endswith('.html')]
    work_ids = []
    work_regex = r'work\d+'

    for file_ in html_files:
        with open(file_, 'r') as f:
            html_code = f.read()
        work_ids_in_file = [work[4:] for work in re.findall(work_regex, html_code)]
        work_ids.extend(work_ids_in_file)
    
    work_ids = list(set(work_ids))

    with open(os.path.join(data_dir, out_file), 'w') as f:
        f.write('\n'.join(work_ids))
