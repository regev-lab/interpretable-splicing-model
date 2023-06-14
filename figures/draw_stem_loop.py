import drawsvg as draw
import numpy as np

def draw_line(d,x1,y1,x2,y2,color):
    d.append(draw.Lines(x1,y1,
            x2,y2,
            close=False,
            fill="none",
            stroke=color,
            stroke_width=2))


def draw_nucleotide(d,x, y, nt, color):
    """ Draw one nucleotide in a circle
    """
    assert(len(nt)==1)
    
    d.append(draw.Circle(x, y, 8,
        fill=color, stroke_width=2, stroke='black'))
    deltax = 4.3 if nt=="A" else (4.5 if nt=="U" else 4.7)
    d.append(draw.Text(nt, 11, x-deltax, y+4, fill="black", font_weight="bold"))  # use 11pt text 
    
def draw_oligo(d,xs, ys, nts, colors):
    lastx = None
    lasty = None
    
    for (x,y,nt,color) in zip(xs,ys,nts,colors): # first draw all the black lines connecting the nucleotides
        if lastx != None:
            draw_line(d,lastx,lasty,x,y,"black")
        lastx, lasty = x,y

    for (x,y,nt,color) in zip(xs,ys,nts,colors):
        draw_nucleotide(d,x,y,nt,color)

        
BASE_PAIR_COLOR = "#ff4e4e"
DELTA_X = 20
DELTA_Y = 20
LOOP_RADIUS_PER_LOOP_LENGTH = {3:18, 4:20, 5:23,6:26}

def draw_stem_loop(nts, stem_length, colors, filename):
    """ First nucleotide is just upstream of stem; then stem; then loop; then 3' part of stem; then another nucleotide outside stem
    """
    assert(len(nts)==len(colors))
    
    d = draw.Drawing(500, 200, origin='center')
       
    for i in range(stem_length):
        draw_line(d,i*DELTA_X,0,i*DELTA_X,DELTA_Y,BASE_PAIR_COLOR)
    
    loop_length = len(nts) - 2*stem_length - 2   
    assert((loop_length >= 3) and (loop_length <= 6))
    loop_radius = LOOP_RADIUS_PER_LOOP_LENGTH[loop_length]

    
    xs = [-0.7*DELTA_X,] + [DELTA_X * i for i in range(stem_length)] + [DELTA_X * (stem_length-1) + loop_radius* np.cos(2*np.pi*(1/2)/(loop_length+2)) - loop_radius * np.cos(2*np.pi*(i+3/2)/(loop_length+2)) for i in range(loop_length)] + [DELTA_X * (stem_length-1-i) for i in range(stem_length)] + [-0.7*DELTA_X]
    ys = [-0.7*DELTA_Y,] + [0] * stem_length + [DELTA_Y/2 - loop_radius * np.sin(2*np.pi*(i+3/2)/(loop_length+2)) for i in range(loop_length)] + [DELTA_Y]*stem_length + [1.7*DELTA_Y]
    
    draw_oligo(d,xs,ys,nts,colors)
    
    d.set_pixel_scale(2)  # Set number of pixels per geometry unit
    d.save_svg(filename)
    
