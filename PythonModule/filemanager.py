def fileEdit(In,Out,Conv,header = None):
    with open(In) as f_in, open(Out, 'w') as f_out:
        # Transform the rest of the lines
        if header:
           f_out.write(header)
        for line in f_in:
            f_out.write(Conv(line))
            
def fileRead(In,offset = 1):   
    x = list()
    y = list()

    with open(In) as f_in:
        for i in range(offset):
            header = f_in.readline()
        for line in f_in:
            data = line.split(sep=",")
            x.append(float(data[0]))
            y.append(float(data[1].strip()))
    return [x,y,header]