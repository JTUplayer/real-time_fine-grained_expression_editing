import PySimpleGUI as sg
import cv2
import numpy as np
from models.stylegan2.model import Generator
import torch
class decoder:
    def __init__(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.decoder = Generator(1024, 512, 8).cuda()
        self.decoder.load_state_dict(ckpt, strict=True)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((512, 512))
    def get_keys(self,d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    def generate(self,codes):
        images, result_latent = self.decoder([codes], input_is_latent=True,randomize_noise=True, return_latents=False)
        images = self.face_pool(images)
        images=(images+1)*127.5
        return images
@torch.no_grad()
def main():
    sg.theme('LightGreen')
    net = decoder('StyleGAN.pt')
    AU_direction=[]
    for i in range(17):
        AU_direction.append(torch.from_numpy(
            np.load(
                'checkpoint/AU-%d.npy' % i)).cuda().unsqueeze(
            0))
    selected_aus=[1, 2, 3, 5, 6, 8, 10, 16]
    au_name = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
    select_id = [17, 18, 37, 55,
                 9, 72, 85, 94,
                 106, 388, 186, 241,
                 298, 340, 116, 524,
                 622, 1123, 2000, 2462,
                 2565, 2740, 3066, 15, ]
    left = [
        # [sg.Text('Demo', size=(60, 1), justification='center')],
        [sg.Button("", size=(0, 0), visible=False)],
        [sg.Push(), sg.Button('Reset', pad=((0, 30), 0), font=('Arial', 15)),
         sg.Button('Next Image', font=('Arial', 15)), sg.Push()]
    ]
    for selected_au in selected_aus:
        left.append([sg.Text('AU %d' % au_name[selected_au],
                             pad=((0, 10), (20, 0)) if au_name[selected_au] < 10 else ((0, 0), (20, 0))),
                     sg.Slider((-5, 5), 0, 0.01, orientation='h', size=(40, 15), key='-AU %d-' % selected_au)])

    left.append([sg.Push(), sg.Button('Exit', size=(10, 1), pad=(0, (20, 0)), font=('Arial', 15)), sg.Push()])
    right = [[sg.Image(filename='', key='-IMAGE-')]]
    layout = [
        [sg.Column(left),
         sg.VSeperator(),
         sg.Column(right), ]
    ]
    # create the window and show it without the plot
    window = sg.Window('Demo', layout)

    latents = np.load('latents/latents.npy')
    latents = torch.from_numpy(latents).cuda()
    last_AU=np.zeros(17)
    w = latents[select_id[0]:select_id[0]+1]
    event, values = window.read(timeout=20)
    frame = np.array(net.generate(w).cpu())[0]
    frame = frame.transpose((1, 2, 0))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    window['-IMAGE-'].update(data=imgbytes)
    tmp_id=0
    while True:
        edit = False
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            break
        if event=='Reset':
            for i in selected_aus:
                window['-AU %d-'%i].update(value=0)
        if event=='Next Image':
            for i in selected_aus:
                window['-AU %d-'%i].update(value=0)
            tmp_id+=1
            edit=True
        w=latents[select_id[tmp_id]:select_id[tmp_id]+1]
        for i in selected_aus:
            if values['-AU %d-'%i]!=last_AU[i]:
                last_AU[i]=values['-AU %d-'%i]
                edit=True
        if edit:
            delta_w=0
            for i in selected_aus:
                delta_w+=values['-AU %d-'%i]*AU_direction[i]
            frame=np.array(net.generate(w+delta_w).cpu())[0]
            frame=frame.transpose((1,2,0))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['-IMAGE-'].update(data=imgbytes)

    window.close()
main()