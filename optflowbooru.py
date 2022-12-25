import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  
    def __init__(self):
        super().__init__()
        self.pre_img_cv=None

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "OptFlowBooru(batch_i2i)"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        deepbooru_limits=gr.Slider(label="deepbooru limits per one image", minimum=0, maximum=30, step=1, value=10)
        muted_regex_str = gr.Textbox(label="Mute word(Regex)", lines=1, value='blur(:?rry)?|lowres|out of frame')
        token_limits=gr.Slider(label="OptFlowbooru Token limits ", minimum=0, maximum=100, step=1, value=30)

#        token_limits = gr.Slider(label="DeepbooruTokenLimits", minimum=1, maximum=150, step=1, value=75)
#        return [deep_str,token_limits]
        return [deepbooru_limits,muted_regex_str,token_limits]

  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p, deepbooru_limits,muted_regex_str,token_limits):
        deepbooru_limits=int(deepbooru_limits)
        token_limits=int(token_limits)

        # optflowして、エッジとって、ハフ変換（円検出）で「あたり」を取ったイメージ群を返す。
        def optflow_images(img_pil, hint_draw=None):
            import cv2
            import numpy as np
            import math

            # 参考にしたページ　https://x1freeblog.com/%E3%80%90python%E3%80%91pillow%E5%9E%8B%E3%81%A8opencv%E5%9E%8B%E3%81%AE%E5%A4%89%E6%8F%9B%E6%96%B9%E6%B3%95%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/951/#toc4
            ### PIL型 => OpenCV型　の変換関数
            def pil2opencv(img_pil):
                out_image = np.array(img_pil, dtype=np.uint8)

                if out_image.shape[2] == 3:
                    out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
                return out_image

            img_cv = cv2.cvtColor(pil2opencv(img_pil),cv2.COLOR_BGR2GRAY)
            h, w= img_cv.shape
            size = (w+h)/2

            if self.pre_img_cv is None:
                self.pre_img_cv=img_cv
                return [img_pil]

            if self.pre_img_cv.shape != img_cv.shape:
                self.pre_img_cv=img_cv
                return [img_pil]

            #https://code-graffiti.com/opencv-dense-optical-flow-in-python/#toc1
            #calcOpticalFlowFarneback(prevImg, nextImg, flow, pyrScale, levels, winsize, iterations, polyN, polySigma, flags）
            flow = cv2.calcOpticalFlowFarneback(self.pre_img_cv,img_cv, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow=np.clip(flow * 255, a_min = 0, a_max = 255).astype(np.uint8)
            edge=cv2.Canny(flow, 200,250) # Hough変換内でエッジ取る処理しているらしいが一応。

            # パラメータ、ユーザーにどこまでいじらせるかね。悩みどころ。
            circles = cv2.HoughCircles(edge,cv2.HOUGH_GRADIENT,1,minDist=int(size/5/2),
                            param1=50,param2=30,minRadius=int(size/5),maxRadius=int(size))

            img_pil_list=[img_pil]

            if not (circles is None):
                for i in circles[0,:]:
                    left=int(i[0]-i[2])
                    upper=int(i[1]-i[2])
                    right=int(i[0]+i[2])
                    lower=int(i[1]+i[2])
#                    print((left, upper, right, lower))
                    img_pil_list.append(img_pil.crop((left, upper, right, lower)))

                    if not(hint_draw is None):
                        hint_draw.rectangle((left, upper, right, lower), fill=None, outline=None)
                        

            self.pre_img_cv=img_cv

            return img_pil_list

        # deepbooruを一時的にハイジャックして、細かいとこまで拾えるようにする。
        # 毎回モデルを出し入れしているので、設定でkeep vram推奨か？
        def deepbooru(img):
            from modules.deepbooru import model as deepbooru_model
            from modules import shared
            deepbooru_model.start()

            tmp_interrogate_return_ranks=shared.opts.interrogate_return_ranks
            shared.opts.interrogate_return_ranks=True

            tmp_interrogate_deepbooru_score_threshold=shared.opts.interrogate_deepbooru_score_threshold
            shared.opts.interrogate_deepbooru_score_threshold=0.05

            tmp_deepbooru_use_spaces = shared.opts.deepbooru_use_spaces
            shared.opts.deepbooru_use_spaces=True

            tmp_deepbooru_escape = shared.opts.deepbooru_escape
            shared.opts.deepbooru_escape=False

            tmp_deepbooru_sort_alpha = shared.opts.deepbooru_sort_alpha
            shared.opts.deepbooru_sort_alpha=True

            deep_prompt=deepbooru_model.tag_multi(img, False)

            shared.opts.interrogate_return_ranks=tmp_interrogate_return_ranks
            shared.opts.interrogate_deepbooru_score_threshold = tmp_interrogate_deepbooru_score_threshold
            shared.opts.deepbooru_use_spaces=tmp_deepbooru_use_spaces
            shared.opts.deepbooru_escape=tmp_deepbooru_escape
            shared.opts.deepbooru_sort_alpha=tmp_deepbooru_sort_alpha

            deepbooru_model.stop()
            return deep_prompt


#        from modules import images
#        from PIL import Image, ImageDraw
#        hint_img=p.init_images[0].copy()
#        hint_draw=ImageDraw.Draw(hint_img)
#        img_pil_list=optflow_images(p.init_images[0], hint_draw)

        img_pil_list=optflow_images(p.init_images[0])

        img_pil_list=img_pil_list[0:deepbooru_limits]

        import re

        deep_prompt_dict={}

        for img_pil in img_pil_list:
            deep_prompt=deepbooru(img_pil)
            for wc in re.split(',\s*', deep_prompt):
                m=re.match('^\((.+):([0-9\.]+)\)$', wc)
                word=m.group(1)
                rank=float(m.group(2))

                # 面積に反比例したウェイトを入れる。細かいディティールを拾えることを期待して。
                # 面積が0ならカウントしない。0徐算を防ぐためにも。
                if img_pil.width!=0 and img_pil.height!=0:
                    rank/=1.0*img_pil.width*img_pil.height
                else:
                    rank=0.0

                # 辞書に単語がなかった・あった場合
                if word in deep_prompt_dict:
                    None
                else:
                    deep_prompt_dict[word]=0.0
    
                # ミュートする（した）単語を除いてカウントする。
                if re.search(muted_regex_str,word) is None: # muted word
                    #print("word muted")
                    None
                else:
                    deep_prompt_dict[word]+=rank

        # カウントに基づいて辞書をソートする。
        deep_prompt_list = sorted(deep_prompt_dict.items(), key = lambda k : k[1])

        # プロンプトのトークン数の足切り
        deep_prompt_list=deep_prompt_list[0:token_limits]

        deep_prompt=",".join(list(map(lambda x: x[0], deep_prompt_list)))
        print(f'additional prompt from flowbooru:{deep_prompt}')

        p.prompt+=deep_prompt

        processed = process_images(p)

# なんかヒントのイメージをweb-uiに表示させたいが、なんかよくわかんなくて、できないので、後でやる。
#        diff_img=images.image_grid([hint_img,processed.images[0]], batch_size=1, rows=None)
#        diff_img = Image.new('RGB', size=(processed.images[0].width*2, processed.images[0].height), color='black')
#        diff_img.paste(hint_img,(0,0))
#        diff_img.paste(processed.images[0],(processed.images[0].width,0))
#        p.do_not_save_samples = True
#        p.do_not_save_grid = True
#        #webui side output
#        processed = Processed(processed, [diff_img])

        return processed
