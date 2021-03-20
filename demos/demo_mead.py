import os, sys
import argparse
from tqdm import tqdm
import tarfile
import moviepy.editor as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

def main(args):
    mead_data = args.data
    device = args.device
    savefolder = args.savefolder

    t = tarfile.open(mead_data)
    for member in t.getmembers():
        if 'front/' in member.name and '.mp4' in member.name:
            t.extract(member, '/content')
            clip = mp.VideoFileClip(f'/content/{member.name}')
            audio_path = os.path.join(savefolder, member.name.replace('.mp4', '.wav', 1)[len('video/front/'):])
            clip.audio.write_audiofile(audio_path)
    
    testdata = datasets.TestData('/content/video/', iscrop=True, face_detector=args.detector)

    deca_cfg.model.use_tex = False
    deca = DECA(config = deca_cfg, device=device)
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename'][len("video/front/"):]
        images = testdata[i]['image'].to(device)[None,...]
        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict) #tensor
        deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
    print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('--data', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--savefolder', default='/content/output', type=str)
    parser.add_argument('--detector', default='fan', type=str)

    main(parser.parse_args())