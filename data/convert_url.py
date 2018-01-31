rf = open('iqiyi_url.txt', 'r')
wf = open('iqiyi_download.sh', 'a')
lines = rf.readlines()
for i, line in enumerate(lines):
    new_line = 'you-get --format=SD ' + line.strip() + ' --output-filename=%05d.mp4\n'%(i)
    wf.write(new_line)
wf.close()
rf.close()
