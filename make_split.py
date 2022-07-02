content = ""

for i in range(1,446):
    scene = "2011_09_26/2011_09_26_drive_0009_sync"
    num = str(i)
    side = "l" 
    content = content+scene+" "+num+" "+side+"\n"  

with open("/content/monoseg/splits/scene_09/train_files.txt","w") as f:
    f.write(content)
f.close()