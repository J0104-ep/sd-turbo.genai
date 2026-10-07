import torch # Ejecuta operaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image #diffusers es una libreria espacializada en modeles generativos

print("Cargando el modelo...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype = torch.float32 #float16 o float32
    #indica la precisión numerica con que se cargan los pesos (decimales, la precision para generar la imagen)
)
modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear ")
negative_prompt = "worst quality, low quality, bad quality, low resolution, blurry, distorted, deformed body, bad anatomy, deformed limbs, badly drawn hands, extra fingers, missing fingers, text, watermark, signature, logo, compression artifacts"

print("Generando imagen...")

imagen = modelo(
    prompt=prompt, #Instrucciones o detaller para crear la imagen
    negative_prompt=negative_prompt,
    num_inference_steps=10, #numero de veces que va a tratar de crear o refinar la imagen
    guidance_scale=2.0, #Que tan fiel debe ser al prompt(Creatividad, temperatura)
    width= 1024,
    height=1024
    
).images[0]

imagen.save("imagen.png")

print("imagen guardada.")

