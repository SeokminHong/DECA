import os, sys
import argparse
import shutil
from tqdm import tqdm
import tarfile
import moviepy.editor as mp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg
from decalib.utils import util

def main(args):
    mead_data = args.data
    device = args.device
    savefolder = args.savefolder

    t = tarfile.open(mead_data)
    for member in t.getmembers():
        if 'front/' in member.name and '.mp4' in member.name:
            t.extract(member, '/content')
            video_name = f'/content/{member.name}'
            if args.gen_audio:
                clip = mp.VideoFileClip(video_name)
                audio_path = os.path.join(savefolder, member.name.replace('.mp4', '.wav', 1)[len('video/front/'):])
                util.check_mkdir(os.path.dirname(audio_path))
                clip.audio.write_audiofile(audio_path)
            testdata = datasets.TestData(video_name, iscrop=True, face_detector=args.detector)
            os.remove(video_name)

            deca_cfg.model.use_tex = False
            deca = DECA(config = deca_cfg, device=device)
            for i in tqdm(range(len(testdata))):
                name = testdata[i]['imagename'][len("video/front/"):]
                images = testdata[i]['image'].to(device)[None,...]
                codedict = deca.encode(images)
                opdict, visdict = deca.decode(codedict) #tensor
                deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict, simple=True)
            shutil.rmtree(video_name.split('.')[0])
            print(f'-- please check the results in {savefolder}')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('--data', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--savefolder', default='/content/output', type=str)
    parser.add_argument('--detector', default='fan', type=str)
    parser.add_argument('--gen-audio', type=str2bool, nargs='?', const=True, default=False)

    main(parser.parse_args())