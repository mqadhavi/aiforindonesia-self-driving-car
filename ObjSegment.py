import streamlit as st
from PIL import Image
import io
import onnxruntime
from torchvision import transforms
import numpy as np
from PIL import Image
from matplotlib import cm

obj = {
    0:"sky",
    1:"building",
    2:"pole",
    3:"road",
    4:"sidewalk",
    5:"nature",
    6:"sign",
    7:"fence",
    8:"car",
    9:"person",
    10:"rider",
    11:"bicycle"
}

def detect_obj(mask_array):
   unique_classes = np.unique(mask_array)
   detected = [obj[key] for key in obj if key in unique_classes]
   return detected
   
def to_numpy(_tensor):
   return _tensor.detach().cpu().numpy() if _tensor.requires_grad else _tensor.cpu().numpy()

def process_image(img):
   transform = transforms.Compose([
      transforms.Resize((512,512)),
      transforms.ToTensor()
   ])
   return transform(img).unsqueeze(0)

def cmap(pre_mask):
   vmin, vmax = 0, 11
   image_scaled = np.clip((np.array(pre_mask) - vmin) / (vmax - vmin), 0, 1)
   viridis_mapped = cm.viridis(image_scaled)
   mask = Image.fromarray((viridis_mapped * 255).astype(np.uint8))
   return mask

def predict_seg(input):
   input = Image.open(input).convert("RGB")
   width, height = input.size
   img = process_image(input)

   ort_session = onnxruntime.InferenceSession("UNet_ResNet_CityScape.onnx", providers=["CPUExecutionProvider"])
   ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
   ort_outs = ort_session.run(None, ort_inputs)
   out = np.squeeze(np.array(ort_outs))

   pred_labels = np.uint8(np.argmax(out, axis=0))
   pred_labels_pil = transforms.ToPILImage()(pred_labels).convert('L')
   detected = detect_obj(np.array(pred_labels_pil))
   resize = transforms.Resize((height, width), interpolation=Image.NEAREST)
   mask = cmap(resize(pred_labels_pil))
   return Image.blend(input.convert('RGBA'), mask, 0.8), detected

st.set_page_config(page_title="My App", layout="wide")
st.title("UNet(with ResNet backbone) not so Self Driving Segmentation")

uploaded_file = st.file_uploader("Upload an image or video\n", type=['png', 'jpg', 'jpeg'])
if uploaded_file is not None and (uploaded_file.type == "image/jpeg" or uploaded_file.type == "image/png"):
   col1, col2= st.columns(2)
   col1.header("Original Image")
   col1.image(uploaded_file, use_column_width=True)

   pred_img, detected = predict_seg(uploaded_file)
   col2.header("Predicted Masks")
   col2.image(pred_img, use_column_width=True)

   st.text("Detected objects:")
   st.text(", ".join([item for item in detected]))

   img_byte_array = io.BytesIO()
   pred_img.convert('RGB').save(img_byte_array, format='JPEG')
   img_byte_array = img_byte_array.getvalue()

   if st.download_button(label="Download Image", data=img_byte_array, file_name='detected_masks.jpg',
                        mime='image/jpeg'):
      st.success("Downloaded successfully!")