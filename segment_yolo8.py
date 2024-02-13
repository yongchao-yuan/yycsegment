"""
SegementPicture based on yolov8
"""
import streamlit as st
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import cv2
import os
from utils import save_mask_pic, get_yolo8_class
from PIL import Image

st.set_page_config(page_title="picture segmentation", page_icon="🙍")
st.markdown("# picture segmentation")


def run_app():
    st.markdown("### 第一步：选择本地的一张图片(png/jpg)...")
    uploaded_file = st.file_uploader(" ")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='上传的图片',
                 use_column_width=True)
        image = Image.open(uploaded_file)
        # 保存图片到服务器指定目录
        org_picname = os.path.join("pics", uploaded_file.name)
        image.save(org_picname)
        st.markdown("### 第二步：设置相关参数")
        wholeflag = st.checkbox('是否整体抠图', value=False, key="whole_ck")
        clslst = st.multiselect(
            "请选择识别的对象类型", get_yolo8_class(-2), "person", key="cls_ms")
        bkname_dict = {"透明": None, "白色": (255, 255, 255), "红色": (0, 0, 255), "蓝色": (255, 0, 0), "绿色": (0, 255, 0),
                       "灰色": (128, 128, 128), "黑色": (0, 0, 0)}
        bkname = st.selectbox("请选择背景颜色", list(
            bkname_dict.keys()), 0, key="bkcolor_s")
        st.markdown("### 第三步：抠图")
        if st.button("抠图"):
            flst = save_objs_png(org_picname, clslst, "pics", wholeflag,
                                 bkname_dict[bkname])
            for f in flst:
                st.image(f, caption=f, use_column_width=True)


def save_objs_png(pathname, clslst, savedir, wholeflag=False, bkcolor=None):
    model = YOLO('models/yolov8s-seg.pt')
    res = model.predict(pathname, retina_masks=True)
    file_num = 0
    fname_lst = []
    for r in res:
        img = np.copy(r.orig_img)
        img_name = Path(r.path).stem
        file_num += 1

        if wholeflag == False:
            cnt = 0
            for ci, c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]
                if label not in clslst:
                    continue

                cnt += 1
                iso_crop = save_mask_pic(img,
                                         [c.masks.xy.pop()],
                                         [c.boxes.xyxy.cpu().numpy().squeeze()],
                                         background=bkcolor)

                # TODO your actions go here
                fname = os.path.join(savedir, f"{img_name}{label}{ci}.png")
                cv2.imwrite(fname, iso_crop)
                fname_lst.append(fname)
        else:
            ldict = {}
            mask_lst = []
            box_lst = []
            for ci, c in enumerate(r):
                label = c.names[c.boxes.cls.tolist().pop()]
                if label not in clslst:
                    continue

                if label not in ldict:
                    ldict[label] = 1
                else:
                    ldict[label] += 1

                mask_lst.append(c.masks.xy.pop())
                box_lst.append(c.boxes.xyxy.cpu().numpy().squeeze())

            if len(ldict) > 0:
                iso_crop = save_mask_pic(
                    img, mask_lst, box_lst, background=bkcolor)
                # TODO your actions go here
                lstr = ""
                for key in ldict.keys():
                    lstr += key+str(ldict[key])
                fname = os.path.join(savedir, f"{img_name}{lstr}.png")
                cv2.imwrite(fname, iso_crop)
                fname_lst.append(fname)
    return fname_lst


if __name__ == "__main__":
    run_app()
