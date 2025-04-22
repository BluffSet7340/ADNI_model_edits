# Model Performance Summary

## **Inference Accuracy**
| Model          | Accuracy |
|---------------|----------|
| **AlexNet**   | 25.15%   |
| **ResNet18**  | 21.79%   |
| **EfficientNetB3** | 30.36% |
| **VGG16**     | 29.29%   |
| **InceptionV3**  |  29.68% |
| **ViT**  |  27.22% |
| **EfficientNetB7** | 17.43% |

---

## **Training Details (ResNet18 - With UnderSampling | Adam Optimizer | LR= 0.0001 | 5 Fold CV 10 Epochs)**
| Model        | Average Train Accuracy | Average Val Accuracy |
|--------------|------------------------|----------------------|
| ResNet18     | 98.22%                 | 77.62%               |

---

## **Next Steps**
- Investigate **dataset balancing strategies** further.
- Consider **data augmentation** to improve generalization.
- Experiment with **learning rate schedules** to stabilize training.
- **Fine-tune pre-trained models** on the dataset instead of training from scratch.

---

## Model Training and Testing Conducted via 5-Fold Cross Validation

## Model Performance for T1_augmented_hflip (horizontal flips, 900 ~ samples per class)
| ViT           | 97.51%                    | 86.13%                            | 15                       | 0.0001            |
Due to the computationally expensive nature of cross validation, in our case 5 fold cross validation, we implemented a an early stopping where if the val_loss doesn't improve after 5 epochs, it breaks and moves on to the next fold. We used the Adam Optimizer with a LR of 0.0001

| Model         | Average Training Accuracy | Average Validation Accuracy       | Epochs Per Fold          | **Learning Rate** |
|---------------|---------------------------|-----------------------------------|--------------------------|-------------------|
| AlexNet       | 97.88%                    | 92.84.%                           | 15                       | 0.0001            |
| InceptionV3   | 98.43%                    | 93.55%                            | 15                       | 0.0001            |
| VGGNet16      | 99.42%                    | 91.93%                            | 15                       | 0.0001            |
| ResNet18      | 99.17%                    | 93.39%                            | 15                       | 0.0001            |
| EffcientNetB3 | 98.74%                    | 93.10%                            | 15                       | 0.0001            |
| EffcientNetB7 | 98.61%                    | 92.94%                            | 15                       | 0.0001            |

----------------------

Results of Using InceptionV3 on the augmented-images-v3 dataset with various hyperparameters - 

With a learning rate of 0.001 the use of the Scheduler, we implemented a reduce learning rate on plateau where if the validation loss didn't improve after 3 epochs, the learning rate is then halved by a factor of 0.5. We implemented this scheduler for a variety of models and found that InceptionV3 performed the best, giving us an average validation accuracy of 71.17% and average training accuracy of 99%. Since this is a clear sign of overfitting, where the training accuracy far exceeds the validation accuracy, we turn to further fine tuning of the model.

We implemented early stopping into the InceptionV3 where if the validation loss doesn't improve after 5 epochs, early stopping would be triggered. In our case, the model scored an 60.89% for average validation accuracy and 91.51% for average training accuracy. The average validation accuracy is slightly worse than the reduce learning rate on plateau scheduler, thus requiring further fine tuning.

Modifying the final fully connected layer to include the following - 

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),  # Reduce features from 2048 to 512
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),  # Further reduce features to 256
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, len(dataset.classes))  # Final layer for 3 classes
    )

This was done to make the model less complex to adapt it to our dataset. However, with the early stopping implemented,
it yielded mediocre results, 62.52% average validation accuracy and 89.78% average training accuracy. Adding another Linear layer with 2048 inputs and 1024 outputs also didn't yield much improvement, giving us an average validation accuracy of 62.66% and 83.96% average training accuracy.

Initially, we assumed that the issue with the MRI images lied in the presence of the skull, eyes, and other irrelevant tissues when it comes to the classification of AD, CN, and MCI. Skull stripping involves the removal of non-brain tissue in MRI scans. Skull stripping has the advantage of removing tissue that isn't affected by Alzheimer's disease, which can potentially throw off the predictions of the pre-trained models, as they might focus more on the skull rather than the brain tissue, as suggested by [Impact of Skull Segmentation in MRI images for Alzheimer’s] paper 

A paper from Harvard student [citation required] detailed the implementation of SynthStrip, a tool that uses deep learning to remove the skeleton portion of the mri images, which has resulted in improvement in model performance (cite a paper). However, despite applying the skull stripping, InceptionV3 still performed poorly, with an average validation accuracy of 50.16% and average training accuracy of 72.80%. This was done via Early Stopping with a patient value of 8 epochs.

One potential issue discovered in the dataset is the fact that despite selecting the middle axial slice of each .nii file, the images tend to vary somewhat. This occured despite selecting the same MRI description for each .nii file, ensuring consistent preprocessing of each file. To cope this, standardization and normalization of the nifti files had to be performed, first starting with the orientation of the
nifti file to standard space via the FSL [citation] library. After this,  


Limitation of Subject Wise Splitting [citation here] 
