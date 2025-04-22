import requests
import re
import random
from PIL import Image
from io import BytesIO
# quicklatex renderer API

def render_latex_quicklatex(latex_code, filename=None):
    img_url, error = quicklatex_render(
        formula=latex_code,
        fontsize="20",
        textcolor="000000",  
        bkgcolor="FFFFFF",   
        preamble=r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}",
        showerrors=True
    )
    if img_url:
        # print("Success! Image URL:", img_url)
        image = download_image(img_url, filename)
        return image
    else:
        print("Failed to render:", error)
        return latex_code
    
def quicklatex_render(
    formula,
    fontsize="12",       
    textcolor="000000",  
    bkgcolor="FFFFFF",   
    preamble="",        
    showerrors=False
):
    """
    模拟 quicklatex.com 前端 AJAX 调用，渲染公式并返回图片 URL。
    如果渲染失败，返回 (None, errmsg)。
    """

    # 1) 与官方 JS 一样，替换公式中的 '%' -> '%25'，'&' -> '%26'
    formula = formula.replace("%", "%25").replace("&", "%26")
    preamble = preamble.replace("%", "%25").replace("&", "%26")

    # 2) 拼接 body 字符串，与前端 script 中的 body 一致
    body = f"formula={formula}"
    body += f"&fsize={fontsize}px"
    body += f"&fcolor={textcolor}"
    body += "&mode=0"  # 对应 latexmode=0
    body += "&out=1&remhost=quicklatex.com"
    if preamble:
        body += f"&preamble={preamble}"
    if showerrors:
        body += "&errors=1"
    # 防止缓存
    body += f"&rnd={random.random()*100}"

    # 3) POST 请求
    url = "https://www.quicklatex.com/latex3.f"
    try:
        resp = requests.post(
            url,
            data=body, 
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30
        )
    except Exception as e:
        return None, f"Request failed: {e}"

    if resp.status_code != 200:
        return None, f"HTTP error: {resp.status_code}"

    # 4) 解析返回的文本
    # 官方 JS 里用的 pattern:
    #   /^([-]?\d+)\r\n(\S+)\s([-]?\d+)\s(\d+)\s(\d+)\r?\n?([\s\S]*)/
    text = resp.text.strip()
    pattern = re.compile(r'^([-]?\d+)\r\n(\S+)\s([-]?\d+)\s(\d+)\s(\d+)\r?\n?([\s\S]*)')
    match = pattern.match(text)
    if not match:
        return None, f"No match. Raw response:\n{text}"

    status = match.group(1)  # "0" 表示成功
    imgurl = match.group(2)
    valign = match.group(3)
    imgw   = match.group(4)
    imgh   = match.group(5)
    errmsg = match.group(6)

    if status == "0":
        # 渲染成功
        return imgurl, None
    else:
        # 渲染失败
        return None, errmsg

def download_image(url, filename=None, dpi=(200, 200)):
    resp = requests.get(url)
    print(url)
    if resp.status_code == 200:
        image_data = resp.content
        img = Image.open(BytesIO(image_data))
        rgb_img = img.convert("RGB")
        if filename:
            # rgb_img.save(filename, "PNG")
            # Save with high DPI and no compression
            rgb_img.save(
                filename,
                format="PNG",
                dpi=dpi,
                optimize=True,      # optional: tries to reduce file size losslessly
                compress_level=0    # PNG compression (0 = no compression)
            )
            print(f"Image saved to {filename}")
    else:
        print("Failed to download image:", resp.status_code)
    return rgb_img

def compare_latex(gt, pred, filename):
    gt_img = render_latex_quicklatex(gt)
    pred_img = render_latex_quicklatex(pred)
    total_height = gt_img.size[1] + pred_img.size[1]
    max_width = max(gt_img.size[0], pred_img.size[0])
    new_img = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    new_img.paste(gt_img, (0, y_offset))
    new_img.paste(pred_img, (0, gt_img.size[1]))
    new_img.save(filename, 'PNG')

if __name__ == "__main__":
    # 测试
    # latex_code = r"$$\mathcal { D } _ { \Gamma } ( w ) = \sqrt { \mathrm { B e r } \, \Omega _ { 0 } | _ { \Gamma } } \equiv \sqrt { \frac { \partial _ { r } z ^ { A } } { \partial w ^ { \mu } } \Omega _ { ( 0 ) A B } \frac { \partial _ { t } z ^ { B } } { \partial w ^ { \nu } } } ,$$"
    import json
    with open('lora_v2_evaluation_results.json', 'r') as ef:
        eval_results = json.load(ef)
    with open('lora_v2_inference_results.json', 'r') as infer:
        infer_results = json.load(infer)
    incorrect_sample_ids = eval_results.get('incorrect_sample_ids', [])
    id_dict = {item['id']: item for item in infer_results}
    root = 'vis_latex/'
    for incor_id in incorrect_sample_ids:
        sample = id_dict.get(incor_id, {})
        compare_latex(sample['gt'], sample['pred'], root+str(incor_id)+'.png')

    # img_url, error = quicklatex_render(
    #     formula=latex_code,
    #     fontsize="17",
    #     textcolor="000000",  
    #     bkgcolor="FFFFFF",   
    #     preamble=r"\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}",
    #     showerrors=True
    # )

    # if img_url:
    #     print("Success! Image URL:", img_url)
    #     download_image(img_url, 'test.png')
    # else:
    #     print("Failed to render:", error)
