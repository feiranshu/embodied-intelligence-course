"""Generate an exact-size printable ArUco marker and PNG, without any hardware."""
from pathlib import Path
import cv2
import numpy as np
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from common import write_image


def main():
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    bits=cv2.aruco.generateImageMarker(dictionary,0,6)
    rects=[]
    for row in range(6):
        for col in range(6):
            if bits[row,col]==0:rects.append(f'<rect x="{col+1}" y="{row+1}" width="1" height="1"/>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" width="60mm" height="60mm" viewBox="0 0 9 9" shape-rendering="crispEdges"><rect width="9" height="9" fill="white"/><g transform="translate(.5 .5)" fill="black">'+''.join(rects)+'</g></svg>'
    # Six cells occupy 40 mm of the 60 mm canvas; surrounding white is 10 mm each side.
    (HERE/'marker_id0_40mm.svg').write_text(svg,encoding='utf-8')
    marker=cv2.aruco.generateImageMarker(dictionary,0,400)
    png=np.full((600,600),255,np.uint8);png[100:500,100:500]=marker
    write_image(HERE/'marker_id0_40mm.png',png)
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>打印ArUco ID0</title><style>@page{size:A4;margin:20mm}body{font:16px sans-serif}img{display:block;width:60mm;height:60mm}button{padding:12px}@media print{button{display:none}}</style><h1>ArUco ID 0 · DICT_4X4_50</h1><p>选择实际大小／100%，关闭“适合页面”。黑色正方形应为40 × 40 mm。</p><img src="marker.svg" alt="ArUco ID0"><p>打印后测量黑色外边长，再填写任务三网页参数。白边不计入尺寸。</p><button onclick="window.print()">打印</button></html>'''
    (HERE/'print_marker.html').write_text(html,encoding='utf-8')
    # The webpage route is /marker.svg; local file opening needs a matching relative name.
    (HERE/'marker.svg').write_text(svg,encoding='utf-8')
    print('Marker generated: black square 40 mm, complete canvas 60 mm')


if __name__=='__main__':main()
