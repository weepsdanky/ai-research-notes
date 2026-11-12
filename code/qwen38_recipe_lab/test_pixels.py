"""Independent upstream size/resize/normalize/patch/sampling/timestamp oracles."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.v2 import functional as tvF
from pixels import PixelConfig,PixelProcessor,smart_size,frame_indices,timestamps
from processor_reference import load


class PixelTests(unittest.TestCase):
    def test_alpha_is_composited_on_white_for_pil_and_tensor(self):
        torch.manual_seed(119);processor=PixelProcessor()
        rgba=torch.randint(0,256,(4,7,9),dtype=torch.uint8)
        rgba[3,0,:3]=torch.tensor([0,128,255],dtype=torch.uint8)
        expected=load('backend')['instance'].convert_to_rgb(rgba,input_data_format='channels_first')
        torch.testing.assert_close(processor.rgb(rgba),expected,atol=0,rtol=0)
        image=Image.fromarray(rgba.permute(1,2,0).numpy(),'RGBA')
        expected=load('color')['convert_to_rgb'](image,return_numpy=True)
        np.testing.assert_array_equal(processor.rgb(image).permute(1,2,0).numpy(),expected)

    def test_image_full_preprocessing_matches_primary_methods(self):
        torch.manual_seed(121);c=PixelConfig(patch_size=2,image_min_pixels=64,image_max_pixels=1024)
        processor=PixelProcessor(c);ref=load('image');backend=load('backend')['instance']
        image=torch.randint(0,256,(3,9,13),dtype=torch.uint8)
        size=ref['smart_resize'](9,13,4,64,1024)
        dims=SimpleNamespace(shortest_edge=None,longest_edge=None,max_height=None,max_width=None,height=size[0],width=size[1])
        resized=backend.resize(image[None],dims,resample=tvF.InterpolationMode.BICUBIC)
        normalized=backend.rescale_and_normalize(resized,True,1/255,True,c.mean,c.std)
        expected,gh,gw=ref['instance'].patchify(normalized,c.patch_size,c.merge_size,c.temporal_patch_size)
        actual=processor.image(image)
        torch.testing.assert_close(actual['pixel_values'],expected[0],atol=0,rtol=0)
        self.assertEqual(actual['image_grid_thw'].tolist(),[[1,gh,gw]])
        rgba=Image.fromarray(np.zeros((9,13,4),dtype=np.uint8),'RGBA')
        self.assertEqual(processor.image(rgba)['image_grid_thw'].tolist(),[[1,gh,gw]])

    def test_video_odd_padding_sampling_and_timestamps_match_primary(self):
        torch.manual_seed(123);c=PixelConfig(patch_size=2,video_min_pixels=64,video_max_pixels=1024,max_frames=16)
        processor=PixelProcessor(c);ref=load('video');obj=ref['instance'];obj.fps=2.;obj.min_frames=4;obj.max_frames=16
        frames=torch.randint(0,256,(9,3,9,13),dtype=torch.uint8)
        actual=processor.video(frames,source_fps=24,num_frames=3)
        indices=obj.sample_frames(SimpleNamespace(total_num_frames=9,fps=24),num_frames=3)
        np.testing.assert_array_equal(actual['frame_indices'],indices)
        size=ref['smart_resize'](3,9,13,2,4,64,1024)
        normalized=processor.normalized(frames[indices.tolist()],size)
        expected,gt,gh,gw=obj.patchify(normalized[None],2,2,2)
        torch.testing.assert_close(actual['pixel_values_videos'],expected[0],atol=0,rtol=0)
        self.assertEqual(actual['video_grid_thw'].tolist(),[[gt,gh,gw]])
        self.assertEqual(actual['timestamps'],load('prompt')['instance']._calculate_timestamps(indices,24,2))
        for total,fps,target in ((9,24,2),(200,25,2),(7,None,2)):
            expected=obj.sample_frames(SimpleNamespace(total_num_frames=total,fps=fps),fps=target)
            indices,used=frame_indices(total,fps,c,target_fps=target)
            np.testing.assert_array_equal(indices,expected);self.assertEqual(used,fps or 24)

    def test_smart_size_and_invalid_bounds(self):
        image=load('image')['smart_resize'];video=load('video')['smart_resize']
        for h,w in ((1,100),(101,303),(4096,1024),(11,7)):
            self.assertEqual(smart_size(h,w,32,65536,262144),image(h,w,32,65536,262144))
            self.assertEqual(smart_size(h,w,32,4096,262144,5,2),video(5,h,w,2,32,4096,262144))
        with self.assertRaises(ValueError):smart_size(1,201,32,64,1024)
        with self.assertRaises(ValueError):frame_indices(20,24,PixelConfig(),num_frames=3,target_fps=2)


if __name__=='__main__':unittest.main()
