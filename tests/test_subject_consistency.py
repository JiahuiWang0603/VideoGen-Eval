import unittest
import torch
from metrics.subject_consistency import feature_similarity,dino_transform,extract_features


class SubjectTests(unittest.TestCase):
    def test_identical_features(self):
        score,_=feature_similarity(torch.tensor([[1.,2.,3.]]*3))
        self.assertAlmostEqual(score,1.,places=6)

    def test_scale_invariance(self):
        score,_=feature_similarity(torch.tensor([[1.,0.],[10.,0.]]))
        self.assertEqual(score,1.)

    def test_orthogonal_and_negative(self):
        score,rows=feature_similarity(torch.tensor([[1.,0.],[0.,1.],[-1.,0.]]))
        self.assertEqual(score,0.)
        self.assertEqual(rows[1]['similarity_first'],0.)

    def test_formula(self):
        score,rows=feature_similarity(torch.tensor([[1.,0.],[1.,0.],[0.,1.]]))
        self.assertEqual(score,.5)

    def test_preprocessing(self):
        out=dino_transform()(torch.full((1,3,256,256),255.))
        self.assertEqual(tuple(out.shape),(1,3,224,224))
        torch.testing.assert_close(out[0,:,0,0],(torch.ones(3)-torch.tensor([.485,.456,.406]))/torch.tensor([.229,.224,.225]))

    def test_identical_images_feature_normalization(self):
        model=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        images=torch.ones(2,3,8,8)
        features=extract_features(model,images)
        torch.testing.assert_close(features.norm(dim=-1),torch.ones(2))
        self.assertAlmostEqual(feature_similarity(features)[0],1.,places=6)


if __name__=='__main__': unittest.main()
