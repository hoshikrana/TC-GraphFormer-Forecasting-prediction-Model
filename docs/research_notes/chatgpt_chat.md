So I have a tc primed dataset that has 1.6tb of data fory capstone project but can you give me more information about the dataset like they have imgs or not if has how can I convert them from nc to png or any format that I use to train my dl model and I will provide you with problem statement of the project 

## Problem Statement

Tropical cyclones pose a major threat to India's coastal regions, particularly the Bay of Bengal and Arabian Sea, making timely cyclone monitoring essential for disaster management and early warning. The increasing availability of satellite observations provides an opportunity to automate cyclone analysis using Artificial Intelligence and Machine Learning.

However, integrating multi-source satellite data and reliably identifying, classifying, and predicting cyclone development remains challenging. Existing approaches may also have limited evaluation on North Indian Ocean cyclones.

Therefore, there is a need for an AI/ML-based system that uses multi-source satellite imagery to **identify tropical cyclones, classify their developmental stage and intensity, and predict short-term changes in cyclone intensity**, with particular emphasis on cyclones affecting the Indian region.

## Objective

The objective of this project is to develop a deep learning–based Computer Vision system that uses multi-channel satellite imagery, including infrared, water-vapor, and passive-microwave data, to **identify tropical cyclones, classify their developmental stage and intensity, and predict short-term changes in cyclone intensity**. The system will estimate key parameters such as **maximum sustained wind speed and minimum central pressure**, using globally available cyclone observations and best-track data. Its performance will be evaluated on **North Indian Ocean cyclones**, particularly those over the Bay of Bengal and Arabian Sea, to assess cross-basin generalization. The model will be benchmarked against established datasets and relevant published methods, with the aim of enabling faster, more consistent, and scalable cyclone monitoring for disaster management and early-warning applications in India.
…
Your capstone project is a strong Computer Vision + Deep Learning research project. More importantly, TC PRIMED is potentially a very useful dataset for this problem, because your objective involves combining satellite observations with tropical cyclone intensity information.

Before converting the 1.6 TB dataset into PNG images, we need to understand its structure, variables, timestamps, spatial dimensions, and labels. Otherwise, you could accidentally lose important scientific information during preprocessing.

Let's break this into four parts:

Understanding the TC PRIMED dataset.

Understanding .nc (NetCDF) files and whether they contain images.

Converting satellite data into PNG, TIFF, or NumPy arrays.

Designing a dataset and model pipeline around your problem statement.

1. What exactly is TC PRIMED?
TC PRIMED
TC PRIMED

Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset

Suitable for deep learning research
According to the official TC PRIMED website, the current v01r01 dataset contains:

Property

Information

Total size

Approximately 1.6 TB

Coverage

1987–2025

Tropical cyclones

3,552

Satellite overpasses

242,613

File format

NetCDF (.nc)

Geographic coverage

Global

Source: Official TC PRIMED dataset.

What type of data does it contain?
This is the most important part for your project.

TC PRIMED contains several different types of satellite and environmental observations.

Hsuan-Tien Lin > Programs/Data
A. Passive microwave imagery

Multi-channel microwave brightness temperatures from satellites such as GMI, AMSR2, ATMS, MHS and SSMIS.

These observations help identify precipitation structures and cyclone organization.

Alex Boreham (@cyclonicwx) on X
B. Infrared (IR) imagery

Nearly coincident geostationary satellite infrared brightness temperatures.

Useful for identifying cloud-top temperatures, convection, and cyclone structure.

Cyclone Nargis - NASA Science
C. Precipitation data

NASA GPROF rainfall retrievals, including surface precipitation and other precipitation-related measurements.

Useful for studying rainfall distribution and cyclone development.

Example IMD Radar Data — PyScanCf
D. Precipitation radar

TRMM/GPM radar observations, including reflectivity profiles and precipitation characteristics, when available.

Impact of SST Resolution on WRF Model Performance for Wind Field Simulation in the Southwestern Atlantic | MDPI
E. Environmental data

ERA5 atmospheric fields and derived environmental diagnostics, such as wind, humidity, temperature and pressure-related information.

Cyclone FANI(01B) category 3 US, possible landfall near Puri/India shortly after 48hours(VIDEO)
F. Best-track information

Cyclone position, maximum sustained wind speed, minimum central pressure, basin and developmental level.

This is particularly important because these observations can provide your supervised learning targets.


TC PRIMED
+2
2. Does it actually contain images?
Yes. But there is an important distinction.

Your .nc files contain scientific measurements arranged in multidimensional arrays, not necessarily ordinary JPEG or PNG images.

For example, an infrared variable could contain a matrix like:

I
R
=
[
220
223
225
230
218
221
226
231
215
219
224
228
]
IR= 
​
  
220
218
215
​
  
223
221
219
​
  
225
226
224
​
  
230
231
228
​
  
​
 
These numbers represent measured brightness temperatures, not RGB pixel colors.

A satellite image can be generated by visualizing these values.

There are two major file types in TC PRIMED:

File type

What it contains

Overpass files

Satellite observations associated with an individual satellite overpass

Environmental files

Best-track information and ERA5 environmental fields throughout the cyclone's lifetime

Each cyclone generally has multiple overpass files and one environmental file.


rammb-data.cira.colostate.edu
+1
Satellite Imagery and Tropical Floaters | Tropical Tidbits
PMW 1B and 1C products — GPM-API
The Passive Microwave Neural Network Precipitation Retrieval (PNPR) Algorithm for the CONICAL Scanning Global Microwave Imager (GMI) Radiometer
The official documentation also provides sample NetCDF files and their sizes. For example, a GMI overpass sample is approximately 14 MB, while an environmental sample is approximately 118 MB. 

TC PRIMED

3. How do we convert NetCDF into PNG?
You can use Python.

The libraries I recommend are:

xarray – reading NetCDF datasets and their variables.

netCDF4 – working with NetCDF metadata and dimensions.

numpy – numerical operations.

matplotlib – visualizing satellite measurements.

cartopy – geographic visualization when needed.

pytorch – eventually loading the processed data into your DL models.

Install:

pip install xarray netCDF4 numpy matplotlib cartopy h5netcdf

Step 1: Open a NetCDF file
import xarray as xr

file_path = "sample_cyclone.nc"

ds = xr.open_dataset(file_path)

print(ds)

This gives you information about:

Variables

Dimensions

Coordinates

Attributes

Data types

However, TC PRIMED files contain separate NetCDF groups. You should inspect the groups rather than assuming all variables are in the root dataset.

For example:

ds = xr.open_dataset(
    file_path,
    group="group_name",
    engine="netcdf4"
)

print(ds.data_vars)

The actual group name must come from your file's metadata.

Step 2: Inspect the variables
for name, variable in ds.data_vars.items():

    print("\nVariable:", name)
    print("Dimensions:", variable.dims)
    print("Shape:", variable.shape)
    print("Data type:", variable.dtype)
    print("Attributes:", variable.attrs)

This is important because we need to identify which variables correspond to:

Infrared brightness temperatures

Microwave brightness temperatures

Precipitation

Radar observations

Latitude and longitude

Wind speed and central pressure

Step 3: Visualize a satellite variable
Suppose we have identified an appropriate two-dimensional infrared variable.

import matplotlib.pyplot as plt

variable_name = "YOUR_IR_VARIABLE"

data = ds[variable_name].squeeze()

plt.figure(figsize=(10, 8))

plt.imshow(
    data.values,
    cmap="gray",
    origin="lower"
)

plt.colorbar(label="Brightness Temperature")

plt.title("Tropical Cyclone Infrared Observation")

plt.savefig(
    "cyclone_ir.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

This creates a PNG visualization.

Important: This is illustrative code. We should not assume the variable name, dimensions, or coordinate orientation until we inspect your actual file.

Step 4: Convert to PNG correctly
For training, you should normalize the measurements carefully.

For example:

import numpy as np
import matplotlib.pyplot as plt

data = ds[variable_name].values.squeeze()

data = np.asarray(data, dtype=np.float32)

valid = np.isfinite(data)

if valid.any():

    low, high = np.nanpercentile(data, [1, 99])

    normalized = np.zeros_like(data)

    normalized[valid] = np.clip(
        (data[valid] - low) / (high - low),
        0,
        1
    )

    image = (normalized * 255).astype(np.uint8)

    plt.imsave(
        "cyclone.png",
        image,
        cmap="gray"
    )

This produces a display image, but remember:

PNG conversion can discard scientific precision.

Different channels have different physical meanings and value ranges.

Missing-value masks must be preserved.

Normalization parameters should be determined from training data, not independently from every test sample.

My recommendation: Do not convert your entire 1.6 TB dataset into PNG files. Keep the original NetCDF files and create processed NumPy arrays or PyTorch tensors for model training.

4. Important: Understand the actual NetCDF groups
I checked the official v01r01 documentation. It specifies the following groups in the overpass files:

NetCDF group

Purpose

overpass_metadata

Satellite overpass information

overpass_storm_metadata

Cyclone information associated with the overpass

passive_microwave

Microwave observations

GPROF

Precipitation retrievals

radar_radiometer

Radar and radiometer observations

infrared

Infrared satellite observations

Environmental files have different groups:

NetCDF group

Purpose

storm_metadata

Cyclone metadata and best-track information

overpass_metadata

Overpass information

overpass_storm_metadata

Overpass-associated cyclone information

diagnostics

Environmental diagnostics

cylindrical

Environmental fields on a cyclone-centred cylindrical grid

rectilinear

Environmental fields on a latitude-longitude grid


rammb-data.cira.colostate.edu
This is why I recommend inspecting the groups before writing your preprocessing pipeline.

Try this code to inspect your actual file:

from netCDF4 import Dataset

file_path = "sample_cyclone.nc"

def inspect_groups(group, level=0):

    for name, variable in group.variables.items():

        print(" " * level, name)
        print(" " * level, "Shape:", variable.shape)
        print(" " * level, "Dimensions:", variable.dimensions)

    for name, subgroup in group.groups.items():

        print("\n", " " * level, "GROUP:", name)

        inspect_groups(subgroup, level + 4)


with Dataset(file_path, "r") as nc:

    print("ROOT DIMENSIONS:", nc.dimensions.keys())

    inspect_groups(nc)

This will help us discover the actual variable names, shapes and dimensions without making assumptions.

5. PNG vs NumPy vs TIFF: Which should you use?
This is an important architectural decision.

PNG

For visualization
RAMMB: TC Real-Time: SH052015 - Tropical Cyclone (>=96 kt) BANSI - Archive
Good for displaying satellite observations.

Useful for visual inspection and reports.

Usually converts measurements to 8-bit or 16-bit pixel values.

Can lose scientific precision and metadata.

NumPy (.npy)

Recommended for DL
Preserves numerical measurements.

Supports multi-channel arrays.

Works directly with PyTorch.

Easy to organize into training samples.

Zarr

Useful for large datasets
Supports chunked multidimensional arrays.

Useful for large-scale data processing.

Allows reading selected chunks without loading everything.

GeoTIFF

Useful when you need geospatial raster data with geographic referencing.

For example, imagine one cyclone observation contains three satellite channels:

Channel 1: Infrared.

Channel 2: Microwave brightness temperature.

Channel 3: Precipitation rate.

Your training input could be:

X
∈
R
3
×
256
×
256
X∈R 
3×256×256
 
This means:

3 channels.

256 pixels in height.

256 pixels in width.

Your model can then learn spatial patterns across all three channels.

However, there is one complication: the different satellite instruments don't necessarily have identical spatial grids, resolutions, or observation times. We must align them carefully before combining them into a single tensor.

Also, microwave observations are the anchor of TC PRIMED. Infrared observations are included when sufficiently close in time, rather than being guaranteed for every overpass. 

TC PRIMED
+1

6. How I would approach your capstone
Your problem statement actually contains four different ML tasks.

I would separate them rather than trying to train one enormous model that does everything immediately.

Proposed deep learning architecture
Multi-source satellite observations

IR + Microwave + Precipitation + Environmental fields

Preprocessing and spatial alignment

Quality control · Normalization · Regridding · Missing-data masks

Multi-channel CNN / CNN-based encoder

Extract cyclone spatial features

Task 1

Cyclone identification

Cyclone / non-cyclone

Task 2

Stage classification

Depression / storm / severe stages

Task 3

Intensity estimation

Wind speed + central pressure

Task 4

Intensity prediction

Future intensity change

A possible implementation:

Task

Possible architecture

Cyclone identification

CNN / ResNet

Developmental-stage classification

CNN

Wind and pressure estimation

CNN + regression heads

Short-term intensity prediction

CNN + LSTM or temporal model

We should decide the final architecture after understanding how many usable observations and labels you actually have.

A particularly important research consideration
Your dataset is global, but your project focuses on India.

The basin identifier IO represents the North Indian Ocean, between 40°E and 100°E north of the equator.

This includes both:

Bay of Bengal.

Arabian Sea.

I recommend training on multiple global basins and reserving North Indian Ocean cyclones for a separate geographical generalization experiment.

However, don't randomly split individual satellite images. Multiple observations from the same cyclone could otherwise appear in both training and testing, producing misleading results.

Split by cyclone/storm identity, and consider a separate chronological evaluation as well.


TC PRIMED
+1
7. Good news: TC PRIMED already contains some of your prediction labels
This is particularly interesting for your capstone.

The official documentation identifies these variables in the overpass storm metadata:

Variable

Meaning

Unit

intensity

Maximum sustained wind speed

Knots

central_min_pressure

Minimum central pressure

hPa

development_level

Cyclone developmental category

Categorical

intensity_change

Change in wind intensity

Knots

intensity_change_periods

Prediction time intervals

Hours

Even better, intensity_change includes forward changes at 6, 12, 18 and 24 hours, when valid best-track observations are available.

That means we may not need to independently construct every prediction label from scratch.

There are important caveats:

Some observations have missing targets.

The dataset uses -999 as a missing-value indicator for several variables.

The wind speed represents a one-minute average at 10 metres and is measured in knots.

Forward intensity changes are only calculated where the necessary best-track observations exist.


rammb-data.cira.colostate.edu
+1
This is a significant advantage for your project.

8. Official resources you should study
NOAA has actually provided a learning journey specifically for TC PRIMED. I strongly recommend using it rather than learning the file format entirely from scratch.

NOAA's Okeanos Explorer Journeys Into Pacific Deepwater | PIPAP
NOAA TC PRIMED Learning Journey

It contains practical Jupyter notebooks covering:

Understanding NetCDF files.

Understanding overpass files.

Understanding environmental files.

Composite analysis.

Pixel-based neural networks.

Regridding satellite swath observations.

Open NOAA Learning Journey
Official documentation

TC PRIMED v01r01 technical documentation (PDF)
TC PRIMED products and download information
9. My recommended plan for your 1.6 TB dataset
Don't start processing all 1.6 TB immediately.

I suggest the following workflow:

Phase

Work

1

Dataset inspection and understanding

2

Identify North Indian Ocean cyclones

3

Inspect available IR, microwave and environmental variables

4

Understand labels and missing values

5

Select a manageable subset of cyclones

6

Build preprocessing and alignment pipeline

7

Create PyTorch training samples

8

Train baseline models

9

Develop multimodal and temporal models

10

Evaluate on unseen North Indian Ocean cyclones

One additional consideration: TC PRIMED overpasses are irregular satellite observations, not a uniformly sampled video sequence. We will need to account for that when developing the temporal prediction model.

10. What I need from you next
Before writing the complete preprocessing code, I want to inspect your actual dataset rather than give you generic code that might not work with your particular files.

Please provide these three things:

One sample .nc overpass file – preferably a GMI or other passive microwave observation.

One environmental .nc file – from the same cyclone, if possible.

Your dataset folder structure – a screenshot or copied directory tree showing how the 1.6 TB is organized.

If uploading entire files is difficult, start with their filenames and the output of the group-inspection Python script above.

Once I have those, I can help you build:

A complete NetCDF inspection notebook.

A variable identification and visualization notebook.

IR, microwave and precipitation visualization.

A scientific-data-preserving conversion pipeline.

A cyclone-level metadata and label extraction pipeline.

A PyTorch Dataset and DataLoader.

A proper training, validation and testing split.

My main recommendation: Build a scientific-data preprocessing pipeline first, not an image-conversion pipeline. Your model should learn from actual satellite measurements, not merely attractive PNG visualizations.

We should aim for a research-quality capstone with reproducible experiments and a proper North Indian Ocean generalization evaluation.

Yeah I will stand with you but the main problem is there are so many varieties of variables that we need to take care of and so many different img data types I mean ir and more and also as you proposed different models  and architecture for different types of data I like it but what I mean iinstead of already been developed I want to build my own architectures and increase the level really vast like llm level and also try different approaches for research paper for this project like there will be so many papers on this i wana develop another one but it needs to be on another level I like the process you proposed preprocessing, extraction  , and tasks this is good but I want to make it more like a futuristic scope to showcase if a person uploades a video i am for showcase purposes in video we needs to predict all the possible things from the models like the cyclone place , wind speed intensity and more whatever things we can see in cyclones like a futuristic level of showing in movies we can see that but I want to develop it
Now I understand what you're aiming for. You don't want to build another student project that takes satellite images, passes them through ResNet, and predicts cyclone intensity.

You want to build a complete AI-powered Tropical Cyclone Intelligence System — something that looks like a next-generation meteorological intelligence platform, with original deep learning architectures, multimodal understanding, temporal forecasting, and an interactive visualization environment.

And I think that's a much more interesting direction for your capstone and a potential research paper.

There are actually two separate ambitions in what you've described:

Research ambition: Develop novel architectures and learning strategies that can compete with established approaches, rather than simply implementing existing models.

Product ambition: Build a futuristic interface where someone uploads a satellite video or a sequence of observations and watches an AI system analyze the cyclone, reconstruct its structure, track its movement, estimate its intensity, and forecast its future evolution.

We should design both together, but keep the scientific model development separate from the visualization application.

1. Our proposed project vision
GenCast Raises Bar for AI Weather Prediction - AI CERTs News
CYCLONE INTELLIGENCE SYSTEM
Proposed research and product direction
A multimodal, spatiotemporal AI platform for tropical cyclone understanding, monitoring, and forecasting.

Imagine someone uploads a satellite video or a sequence of satellite frames.

The system should automatically:

Detect whether a tropical cyclone is present.

Locate its center and estimate its geographic coordinates.

Identify its structure, eye, eyewall and spiral rainbands where observable.

Estimate maximum sustained wind speed.

Estimate minimum central pressure.

Classify its developmental stage.

Estimate its direction and speed of movement.

Identify structural changes.

Predict intensity changes over the next 6, 12, 24 and potentially 48 hours.

Generate future cyclone movement and intensity visualizations.

Communicate uncertainty and confidence in predictions.

One important scientific distinction: a video generated from satellite imagery is not necessarily a conventional video. It may be a sequence of different satellite observations with different resolutions, channels, and acquisition times. Our system needs to understand those differences rather than treating every frame as an ordinary RGB video frame.

2. Instead of separate ordinary models, let's develop our own architecture
This is where I want us to be ambitious.

I don't recommend simply creating four unrelated models for four tasks. Instead, we can investigate a unified architecture with specialized components for each satellite modality.

Let's give our proposed architecture a working name:

CycloneFusion-X
Proposed original research architecture · Working name

A multimodal spatiotemporal cyclone intelligence architecture

Short-Term Intensity Prediction of Tropical Cyclones Based on Multi-Source Data Fusion with Adaptive Weight Learning | MDPI
The fundamental idea: different satellite instruments observe different physical properties of the same cyclone. Rather than converting everything into RGB and combining it, we teach the network to understand each physical modality independently before learning their relationships.

A. Modality-specific intelligence encoders
We will have specialized encoders for different types of observations.

MeteoNews: Le typhon "Doksuri" apporte des précipitations extrêmes aux Philippines et à Taiwan (26 juillet 2023)
1. Infrared Intelligence Encoder

Learns:

Cloud-top temperature patterns.

Deep convection.

Eye and eyewall characteristics.

Cloud symmetry and organization.

Research direction: develop a custom hierarchical spatial encoder.

NASA’s TROPICS Mission: Offering Detailed Images and Analysis of Tropical Cyclones - NASA Science
2. Microwave Intelligence Encoder

Learns:

Internal precipitation structures.

Convective organization.

Rainband patterns.

Microwave brightness-temperature signatures.

Research direction: channel-aware cross-spectral feature learning.

Gita | NASA Global Precipitation Measurement Mission
3. Precipitation Intelligence Encoder

Learns:

Rainfall distribution.

Precipitation intensity.

Convective asymmetry.

Rainfall structure around the cyclone center.

Research direction: spatial precipitation-pattern representation.

hurricane models Archives - Insurance Journal
4. Atmospheric Intelligence Encoder

Learns:

Environmental wind fields.

Humidity distribution.

Temperature structure.

Atmospheric conditions surrounding the cyclone.

Research direction: physics-informed spatial representation learning.

These are not four complete independent models. They are specialized feature extractors that feed into a common intelligence system.

B. Cross-modal intelligence fusion
This could become one of our most important research contributions.

Imagine:

Infrared shows very cold cloud tops.

Microwave shows an organized precipitation structure.

Environmental data shows favorable atmospheric conditions.

Each modality provides different evidence.

Our fusion module should learn how these observations complement or contradict one another.

CVPR Poster OmniVec2 - A Novel Transformer based Network for Large Scale Multimodal and Multitask Learning
We can investigate:

Cross-modal attention.

Channel-aware attention.

Adaptive modality weighting.

Missing-modality learning.

Contrastive representation learning.

Physical consistency constraints.

For example, if an infrared observation is missing, the model should not necessarily fail. It should be able to use microwave and environmental observations while acknowledging the reduced information.

This is important because real satellite datasets do not guarantee that every modality is available at every timestamp.

C. Temporal Cyclone Intelligence Engine
This is where the project becomes much more interesting.

A cyclone is not a static image. It is a continuously evolving atmospheric system.

We need an architecture that understands how the cyclone changes over time.

How do hurricanes like Beryl form? | wtol.com
Instead of simply passing consecutive frames through a conventional LSTM, I want us to investigate a custom temporal architecture.

Working name: Cyclone Temporal Evolution Network (CTEN).

Its responsibilities:

Learn how cyclone structure changes between observations.

Track the movement of the cyclone center.

Learn the relationship between structural changes and intensity changes.

Capture long-term temporal dependencies.

Forecast future spatial and physical states.

Possible research components:

Component

Research direction

Spatial memory

Learn persistent cyclone structure

Temporal attention

Identify important historical observations

Motion representation

Learn cyclone displacement and movement

Multiscale temporal learning

Capture rapid and gradual changes

Physical constraints

Encourage meteorologically consistent predictions

We can investigate custom attention mechanisms, recurrent state-space models, and hierarchical temporal representations.

I particularly like the idea of a persistent cyclone memory: the model maintains an evolving representation of the cyclone instead of independently interpreting each image.

D. A unified prediction architecture
Finally, all representations enter a shared prediction system.

CycloneFusion-X: proposed architecture
Multi-source satellite observations

IR · Microwave · Rainfall · Radar · ERA5

IR Encoder
Microwave Encoder
Rainfall Encoder
Atmospheric Encoder
Cross-Modal Fusion Engine

Adaptive attention · Shared representation · Missing modality handling

CTEN — Temporal Evolution Network

Cyclone memory · Motion learning · Temporal dynamics

Shared cyclone representation

Detection

Cyclone probability

Localization

Center coordinates

Classification

Developmental stage

Regression

Wind and pressure

Forecasting

Future intensity

Spatial prediction

Future cyclone structure

This is a research hypothesis, not an established architecture or a guarantee of superior performance. We will need to test each component independently and compare it with established baselines.

3. Let's take it to a much higher level: self-supervised pretraining
I think this is one of the most promising directions for your research.

You have approximately 1.6 TB of data. But the number of observations with complete, reliable labels will be much smaller than the total number of data samples.

We should investigate whether we can teach our architecture to understand cyclone observations before giving it all the prediction tasks.

This is similar in spirit to how language models learn representations from enormous amounts of unlabeled text before fine-tuning for specific tasks.

We could build a Cyclone Foundation Model.

SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery
Potential pretraining objectives:

Learning objective

What the model learns

Masked spatial reconstruction

Reconstruct missing satellite regions

Cross-modal reconstruction

Predict IR features from microwave features, and vice versa

Temporal prediction

Predict the representation of a future observation

Contrastive learning

Recognize observations representing the same cyclone state

Physical representation learning

Learn meaningful relationships between environmental variables

For example:

We give the model:

IR at time T.

Microwave at time T.

Environmental fields at time T.

We mask a portion of the microwave observation and ask the model to reconstruct its representation using the remaining information.

Later, we ask it to predict how the cyclone representation evolves.

This gives us a reusable pretrained encoder.

The same encoder can then be fine-tuned for cyclone identification, intensity estimation, developmental-stage classification and forecasting.

One important distinction: This would be a meteorological foundation model, not an LLM. It would use some related ideas from foundation-model research, but learn physical and spatial representations rather than language.

4. The futuristic application you described
Now let's discuss the product itself.

I don't want the final application to look like a normal ML project with an upload button and four prediction labels.

I envision a Cyclone Digital Intelligence Dashboard.

Yapay Zeka ve Meteoroloji: Hava Tahminlerinde Yeni Bir Çağ Başlıyor | Hava Pusulası
Tensor Cores: Versatility for HPC & AI | NVIDIA
China MAZU AI Weather System to Reach 30 Countries: Fengyun Satellites, AI Models, and Radar Unite as Meteorological Infrastructure Goes Global - Pandaily
Imagine a user uploading a sequence of satellite observations or a supported satellite video.

The system processes the input and generates a live analysis.

A. Live cyclone detection and tracking
Deelan Jariwala (@WxTca) on X
The dashboard displays:

Cyclone center.

Latitude and longitude.

Eye location, if identifiable.

Cyclone movement direction.

Translation speed.

Geographic trajectory.

Detection confidence.

For localization, we should develop a spatial prediction head that outputs center coordinates and uncertainty, rather than just a cyclone/no-cyclone classification.

B. Cyclone intensity monitoring
Simulation of Cyclone Christian and Typhoon Haiyan : CEN : Universität Hamburg
The dashboard displays:

Parameter

Example display

Maximum sustained wind

Model estimate in knots

Minimum central pressure

Model estimate in hPa

Developmental stage

Model classification

Intensity tendency

Strengthening / weakening

Movement speed

Estimated km/h

Movement direction

Estimated bearing

These are illustrative display fields, not actual predictions.

We should also display uncertainty intervals, not just a single number.

C. 3D cyclone reconstruction
This is one of the most visually impressive possibilities.

NASA SVS | Volumetric Renderings of Hurricane Isabel based on the WRF Computational Model: close up with winds
Imagine reconstructing:

Eye and eyewall structure.

Spiral rainbands.

Cloud-top height where observations support it.

Precipitation distribution.

Wind-field estimates.

Atmospheric structure.

However, there is a scientific limitation: a 2D infrared or microwave observation alone cannot uniquely reconstruct the true 3D atmosphere.

We would need additional vertical observations, radar profiles, environmental fields, or a trained reconstruction model. The interface must distinguish directly observed information from inferred or simulated information.

D. Future cyclone simulation
Individual Storm Analysis — tropycal
Suppose the user uploads observations from the last 12 hours.

Our model generates:

Future cyclone-center positions.

Future wind-speed estimates.

Future central pressure estimates.

Predicted intensity evolution.

Possible future precipitation structure.

Forecast uncertainty.

We could visualize future states at:

+6 hours

+12 hours

+24 hours

+48 hours

The 48-hour forecast should be treated as an experimental research objective, not something we assume the dataset supports equally well.

Also, our model's forecasts should initially be presented as research predictions, not official disaster warnings or replacements for IMD forecasts.

E. The interface itself
My suggested technology stack:

Layer

Technology

Deep learning

PyTorch

Data processing

Xarray, NumPy, Zarr, Dask

Backend

FastAPI

Frontend

React / Next.js

2D geographic visualization

MapLibre / deck.gl

3D globe

CesiumJS

3D rendering

Three.js

Experiment tracking

MLflow

Model serving

FastAPI + PyTorch

We can build the research system independently of the frontend. The dashboard will consume model outputs through APIs.

5. Research paper: how do we make an actual contribution?
I want to emphasize something important here.

Using a custom architecture does not automatically make a paper novel. A paper needs a clearly identified research gap, a technically defensible contribution, strong baselines, ablation experiments, and reliable evaluation.

We should not claim that we invented a new transformer or a new foundation model just because we combine several existing techniques.

Instead, I propose investigating five research directions.

Research direction 1
Cross-modal cyclone representation learning

Research question: Can we learn a shared representation across infrared, passive microwave, precipitation and atmospheric observations despite differences in resolution, timing and missing modalities?

Potential contribution:

Modality-specific encoders.

Asynchronous cross-modal fusion.

Missing-modality robustness.

Cross-modal self-supervised learning.

My preferred first research direction.

Research direction 2
Physics-informed cyclone evolution learning

Research question: Can physical constraints improve the consistency of predicted cyclone evolution?

Potential contribution:

Physical consistency losses.

Wind-pressure relationship constraints.

Temporal smoothness where physically appropriate.

Motion-aware forecasting.

Research direction 3
Self-supervised cyclone foundation model

Research question: Can large-scale unlabeled satellite observations produce transferable cyclone representations?

Potential contribution:

Masked observation modeling.

Temporal representation prediction.

Cross-modal contrastive objectives.

Transfer learning across ocean basins.

Research direction 4
Uncertainty-aware intensity forecasting

Research question: Can our model estimate not only future cyclone intensity but also how uncertain that prediction is?

Potential contribution:

Probabilistic forecasting.

Quantile regression.

Calibration.

Uncertainty-aware modality fusion.

Research direction 5
Cross-basin generalization

Research question: How well can a model trained using global cyclone observations understand North Indian Ocean cyclones it has never encountered?

Potential contribution:

Leave-one-basin-out evaluation.

North Indian Ocean holdout.

Domain adaptation experiments.

Few-shot regional adaptation.

I would not try to publish five separate contributions in one paper. We should select one central contribution and use the others as supporting experiments.

My initial recommendation is to combine directions 1 and 3, with direction 5 as a major evaluation objective.

That gives us a coherent research hypothesis:

A self-supervised, cross-modal spatiotemporal representation model can improve tropical cyclone analysis and short-term intensity forecasting under heterogeneous and incomplete satellite observations, particularly when generalizing to the North Indian Ocean.

That is a research question we can actually investigate.

6. The complete development roadmap
I suggest dividing the project into seven major phases.

Phase 1 — Dataset intelligence and scientific understanding

Inspect all NetCDF groups.

Catalogue every variable.

Identify different satellite instruments.

Understand labels, dimensions and timestamps.

Identify North Indian Ocean cyclones.

Build a dataset metadata catalogue.

Our immediate next phase
Phase 2 — Advanced preprocessing infrastructure

Multimodal spatial alignment.

Missing-data handling.

Scientific normalization.

Temporal synchronization.

Cyclone-centred cropping.

Efficient chunked data storage.

Phase 3 — Research baselines

Build simple CNN baselines.

Implement established temporal architectures.

Benchmark intensity estimation.

Evaluate cyclone localization.

Establish reliable metrics and splits.

Phase 4 — Original architecture development

Build modality-specific encoders.

Develop cross-modal fusion.

Develop CTEN.

Introduce physical constraints.

Conduct ablation experiments.

Phase 5 — Cyclone Foundation Model

Self-supervised pretraining.

Masked reconstruction.

Temporal representation learning.

Cross-modal learning.

Transfer learning.

Phase 6 — Forecasting and scientific evaluation

Short-term intensity forecasting.

Movement prediction.

Uncertainty estimation.

Cross-basin evaluation.

North Indian Ocean generalization.

Comparison against published methods.

Phase 7 — Cyclone Intelligence Platform

Satellite video/sequence upload.

Automated analysis.

Interactive tracking.

3D cyclone visualization.

Future-state simulation.

Prediction reports and uncertainty visualization.

7. A few things I want us to be careful about
There are three major challenges.

1. Computational requirements

Your 1.6 TB dataset is not something we should blindly load into RAM or convert into millions of images. Your previously mentioned 16 GB RAM laptop with integrated graphics is suitable for developing the inspection and preprocessing pipeline, but large-scale foundation-model training will require access to stronger GPU compute.

We should design the pipeline to process selected chunks and scale gradually.

2. Temporal data availability

TC PRIMED is not a continuous satellite video dataset. We must investigate how observations are distributed across time and which modalities overlap. We may need complementary geostationary satellite datasets for the continuous-video demonstration.

3. Scientific credibility

A visually impressive 3D cyclone is not necessarily an accurate cyclone prediction. Every displayed parameter should have a clear source:

Direct satellite observation.

Best-track reference.

Model estimate.

Model forecast.

Simulated visualization.

This distinction will make the project much stronger academically.

8. Let's finalize our research direction
I want to establish the fundamental scope before we start writing thousands of lines of preprocessing code.

1. What should be our primary ambition?


Build a Cyclone Foundation Model first.

Develop a novel multimodal architecture first.

Focus primarily on advanced cyclone forecasting.

Build all three progressively into one unified system.
2. Which research contribution interests you most?


Cross-modal fusion and custom attention.

New temporal architecture and cyclone memory.

Physics-informed deep learning.

Self-supervised learning and foundation models.

A combination of these approaches.
3. What should the final demonstration accept?


Satellite image sequences.

Satellite video upload.

Both sequences and video.
4. What level are we targeting?


Advanced university capstone.

Research-paper-quality system.

Publication-oriented research with a potentially reusable foundation model.
One final thought: I don't want us to rush into building the dashboard. The real achievement should be the underlying AI architecture. The dashboard should demonstrate what that architecture has learned and what it can predict.

Our goal should be to build something that is scientifically defensible, technically original, and impressive enough to demonstrate as both a research project and a working AI product.

Your project is promising, but several problems can sink it if you don't handle them early. In rough order of severity:

**1. Scope is too big for one capstone**
- A custom LLM-scale architecture, a foundation model, 3D reconstruction, a futuristic dashboard and a research paper together is a multi-year, multi-person effort.
- *Fix:* split it into a must-have core (detection, intensity regression, stage classification, 24h intensity change on TC PRIMED, evaluated on NIO) and stretch goals (self-supervised pretraining, dashboard polish). Only the core needs to be solid for the paper.

**2. "Video" doesn't exist in this dataset**
- TC PRIMED is a set of satellite overpass snapshots at irregular intervals, often hours apart, from different sensors. It isn't a continuous video.
- *Fix:* for the demo, build a sequence player from a storm's ordered overpasses, and say so honestly. You can also interpolate or animate between frames visually, but don't present it as a real video.

**3. No negative samples**
- TC PRIMED contains only cyclones, so "cyclone vs non-cyclone" can't be trained from it alone.
- *Fix:* sample non-cyclone regions or pre-genesis disturbances, or reframe the task as detecting cyclone center and genesis. Be explicit in the paper about how negatives were built.

**4. Channel mismatch with your problem statement**
- Your objective says IR, water-vapor and microwave. As far as I know, TC PRIMED's infrared is a single window channel, not water vapor. Check the v01r01 documentation to confirm.
- *Fix:* either change the objective to match the available channels, or add water-vapor data from another source (e.g., INSAT-3D/3DR).

**5. Very few North Indian Ocean (NIO) storms**
- NIO has only a handful of storms per year, so your test set may be a few dozen storms, with few strong ones. Results will have wide error bars and be statistically fragile.
- *Fix:* use leave-one-year-out or cross-validation over NIO storms, report confidence intervals, and compare NIO against other basins.

**6. Label quality and inconsistency**
- Best-track wind and pressure come from different agencies with different averaging periods (1-minute vs 3-minute winds), and the NIO labels are noisier. Missing values (-999) are common.
- *Fix:* document the label source and convert consistently. Mask missing targets in the loss rather than dropping or imputing them, and report the label noise floor.

**7. Data leakage**
- Splitting by image puts the same storm in train and test, which inflates results badly. Overpasses from the same storm are highly correlated.
- *Fix:* split strictly by storm, with a separate NIO holdout. Compute normalization statistics on train only.

**8. Class imbalance**
- Strong cyclones (Cat 4–5) are rare, so the model will under-predict them.
- *Fix:* use weighted or focal loss, balanced sampling, and report per-class metrics, not only overall accuracy or RMSE.

**9. Heterogeneous sensors and grids**
- Different instruments have different resolutions, channels, swaths and timing. Naively stacking them is scientifically wrong.
- *Fix:* regrid to a storm-centered grid, keep missing-data masks as extra channels, and add sensor/channel embeddings.

**10. Compute and storage (1.6 TB)**
- You won't process this on a laptop, and PNG conversion would destroy precision.
- *Fix:* subsample first, convert to chunked Zarr/NumPy tensors at fixed storm-centered crops, and use Colab/Kaggle credits or your university's GPUs. Make a small prototype dataset (~20–50 GB) before scaling.

**11. "Own architecture" without a real gap**
- A paper needs a defensible novelty, strong baselines and ablations, not just a custom name. Reviewers will check whether your architecture beats simple baselines.
- *Fix:* pick one clear research question, such as missing-modality-robust fusion or cross-basin generalization to NIO. Implement ResNet and simple baselines first, then add your component and show ablations. Compare with published methods (e.g., Advanced Dvorak Technique, DeepMicroNet-style models).

**12. 3D reconstruction and 48h forecasting**
- 2D imagery can't uniquely define 3D structure, and 48h intensity forecasts are hard even for operational models. Weak results here would hurt credibility.
- *Fix:* keep 3D as a clearly labeled "inferred visualization" in the demo, not a scientific claim. Keep forecasting to 6–24h, where the dataset provides labels.

**13. Safety and framing**
- Presenting predictions as warnings can mislead people.
- *Fix:* label outputs as research predictions, show uncertainty, and don't present it as a replacement for IMD.

**14. Timeline risk**
- Final-year capstones fail most often because of a late start on data work.
- *Fix:* suggested order: (1) inspect files and build a data loader, (2) baseline CNN for intensity, (3) NIO evaluation, (4) your novel component, (5) dashboard last.

My recommendation is to make the paper about **cross-basin generalization to the North Indian Ocean with missing-modality-robust fusion**. It is novel enough, directly matches your problem statement, and is achievable.

Want me to turn this into a phased project plan with a realistic timeline and a paper outline?





Yeah and this not just for north region and one more thing that this whole thing needs to be realistic and that we needs to build and my university is providing gpu access and has about Google drive for storage around 5tb and we can use colab for training purposes also the data of possible I can get from world like only free resources available for everyone



Both your resources and the global scope work. Here's a plan that fits Colab, a university GPU, and 5 TB of Drive.

**Data (all free)**
- **TC PRIMED**: your main source. I believe it's also hosted on NOAA's AWS open-data bucket, so you can download directly. Check the TC PRIMED page for the exact location. Don't download all 1.6 TB; start with a subset (a few hundred storms across all basins, plus every NIO storm).
- **IBTrACS**: free global best-track labels (wind, pressure, position, stage), useful for cross-checking.
- **GridSat-B1 (NOAA NCEI)**: global 3-hourly IR and **water-vapor** imagery since 1980. This fixes the water-vapor gap in your objective and gives you regular time steps, so you can build real sequences and a video-like demo.
- **ERA5 (Copernicus CDS)**: free environmental fields, if you want that modality.
- **Himawari / GOES (AWS open data) and INSAT-3D (MOSDAC, free registration)**: optional, for live-style demos on real Indian-region imagery.

**Storage and compute setup**
- Drive's 5 TB is plenty, but Colab reads thousands of small files from Drive very slowly. Pack samples into large shards (Zarr or .npy), copy a shard set to Colab's local disk at the start of each session, and train from there.
- Colab disconnects, so save checkpoints to Drive every epoch and make training resumable. Use the university GPU for long runs and Colab for experiments.
- Preprocess once into storm-centered crops (e.g. 128×128 or 256×256 per channel) with missing-data masks. Never train directly on raw .nc files.

**Phases (tell me your deadline and I'll fit dates)**
1. **Weeks 1–3: data.** Inspect the files, build the loader, create a 20–50 GB prototype set, and set up a split by storm.
2. **Weeks 4–6: baselines.** ResNet or EfficientNet on IR only, then IR + microwave. Tasks: wind, pressure, stage, 24h change.
3. **Weeks 7–10: your architecture.** Modality encoders, a fusion module, and missing-modality training. Run ablations.
4. **Weeks 11–13: evaluation.** Leave-one-basin-out tests (NIO is one highlighted result), per-class metrics, confidence intervals, and comparison with published results.
5. **Weeks 14–16: demo and paper.** Dashboard last.

**Making it global, not just NIO**
- Train on all basins and report results per basin.
- The paper's story: "Does a multi-modal model trained on other basins transfer to unseen basins, and does it survive missing sensors?" NIO becomes your hardest, most relevant test case, but the work isn't limited to it.

**The futuristic demo, built realistically**
- Use GridSat-B1 sequences (3-hourly) for a smooth storm timelapse.
- Overlay your model's outputs frame by frame: center track, wind, pressure, stage, trend, and a forecast cone with uncertainty.
- Label anything inferred, like 3D views, as "visualization, not observation."
- Stack: PyTorch, FastAPI, and React with MapLibre or CesiumJS.

**Next step:** I can write the data-inspection and download script for the TC PRIMED subset, or draft the detailed phase plan with a paper outline. Which first? And what's your submission deadline?




I have about 20weeks so we can do it as we work really week but we needs to work like a professional and solve this problem statement not for real but as a project and we don't need to predict all but we needs to predict somemote than anything that everbuild about this project ok and also I have doubt as discussed we will be training a pretrained model first as you said but if we re fine-tune on same data would it over fit not learn problems will occur right


No, pretraining and fine-tuning on the same dataset isn't a problem by itself. It's standard practice (MAE, SimCLR and similar methods pretrain on a dataset's images, then fine-tune on the same dataset). Pretraining uses no labels, so the model can't memorize your targets. The risks are elsewhere:

- **Test leakage:** this is the real danger. Keep your test storms (including the NIO holdout) out of pretraining too, even though it's unlabeled. Split by storm first, then pretrain only on train-split storms.
- **Fine-tuning that destroys the pretrained features:** use a low learning rate, layer-wise LR decay, and train a linear probe or frozen encoder first, then unfreeze.
- **Pretraining that doesn't help:** this happens often, so don't assume it works. Always compare against the same model trained from random initialization. If pretraining doesn't win, that's still a reportable result.
- **Pretraining collapse (contrastive methods):** masked reconstruction is simpler and more stable, so I'd use that.

Pretraining pays off here because it can use every overpass, including those with missing labels (-999), w
…
Perfect. We'll use a phase-based development strategy, not a rigid weekly schedule. We have approximately 20 weeks, but each phase will be completed based on its deliverables and validation gates.

I also want to establish one important architectural decision: our future prediction system will have two separate outputs.

Future geographical position: Predict the cyclone's future coordinates at +6, +12, +18 and +24 hours, with uncertainty.

Future satellite visualization: Generate an AI-predicted future infrared frame conditioned on historical observations, predicted location and intensity.

The second output will use GridSat-B1 temporal sequences, not pretend that TC PRIMED itself contains continuous video.

Our finalized project structure
Cyclone Intelligence System
20-week research and engineering project

Global cyclone analysis
Phase 1 — Data Intelligence

Understand, inspect, catalogue and validate all available datasets.

Deliverable: reusable scientific data pipeline.

Phase 2 — Data Engineering

Multimodal alignment, storm-centred crops, temporal synchronization and storage.

Deliverable: training-ready dataset.

Phase 3 — Baseline Intelligence

Establish performance for classification, regression and forecasting.

Deliverable: reproducible baseline experiments.

Phase 4 — Original Architecture

Custom modality encoders, fusion, temporal learning and missing-modality robustness.

Deliverable: CycloneFusion-X experimental architecture.

Phase 5 — Foundation Model Research

Self-supervised pretraining and representation learning.

Deliverable: pretrained cyclone encoder and comparison against random initialization.

Phase 6 — Future Intelligence

24-hour intensity, track forecasting and future infrared frame generation.

Deliverable: future cyclone prediction module.

Phase 7 — Evaluation and Research Paper

Global and basin-wise evaluation, NIO generalization, ablations and uncertainty.

Deliverable: research results and paper.

Phase 8 — Intelligence Dashboard

Sequence player, interactive globe, cyclone tracking and predicted future visualization.

Deliverable: complete demonstration application.

Our engineering rules
I want us to work like an actual research engineering team.

No architecture development before understanding the data.

No training on the entire dataset before validating a smaller prototype.

No custom architecture without a baseline comparison.

No pretraining using held-out test storms.

No claiming improvements without ablation experiments.

Every experiment must have recorded configurations, metrics and checkpoints.

Every phase must have a working deliverable before moving forward.

And regarding your pretrained-model concern: we will split storms first, pretrain only on training storms, and then fine-tune using labeled training samples from those same training storms. Validation and test storms will remain excluded from both stages. This avoids the most important form of leakage.

Let's begin Phase 1
Current phase
Phase 1: Dataset Intelligence and Scientific Understanding

Our first objective is not to train a neural network.

It is to understand exactly what we possess.

I want us to create a master dataset catalogue containing:

Every NetCDF file and its type.

Every variable and its physical meaning.

Available satellite instruments.

Image dimensions and spatial resolutions.

Observation timestamps.

Missing-value percentages.

Cyclone IDs and basin distribution.

Available wind, pressure and intensity labels.

Temporal availability of different modalities.

We will eventually have a dataset inspection notebook that automatically generates this catalogue.

Your immediate next step:

Provide me with your TC PRIMED folder structure or a screenshot of the directory containing the files.

We will start with a few representative files, not the entire 1.6 TB.

One more thing: let's maintain a GitHub repository for the research code, preprocessing, experiments, architecture versions and documentation from the beginning. This will make the final project reproducible and much easier to present.

Our ambition is high, but our execution will be scientific: understand → implement → experiment → evaluate → improve.

User attachment
User attachment
import netCDF4 as nc

file_path = "TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc"

with nc.Dataset(file_path, "r") as root:
    print("--- Root Variables ---")
    for var in root.variables:
        print(f"  {var}: shape {root.variables[var].shape}")
        
    print("\n--- Subgroups ---")
    for group_name in root.groups:
        print(f"Group: {group_name}")
        grp = root.groups[group_name]
        for var in grp.variables:
            print(f"  {var}: shape {grp.variables[var].shape}")    --- Root Variables ---

--- Subgroups ---
Group: overpass_metadata
  time: shape (1,)
  basin: shape (1,)
  cyclone_number: shape (1,)
  season: shape (1,)
  instrument_name: shape (1,)
  platform_name: shape (1,)
  granule_number: shape (1,)
  coverage_fraction: shape (1,)
  env_time: shape (1,)
Group: overpass_storm_metadata
  time: shape (1,)
  storm_latitude: shape (1,)
  storm_longitude: shape (1,)
  intensity: shape (1,)
  intensity_change_periods: shape (9,)
  intensity_change_times: shape (9,)
  intensity_change: shape (9,)
  central_min_pressure: shape (1,)
  development_level: shape (1,)
  distance_to_land: shape (1,)
  storm_speed: shape (1,)
  storm_heading: shape (1,)
  storm_speed_meridional_component: shape (1,)
  storm_speed_zonal_component: shape (1,)
Group: passive_microwave
  time: shape (1,)
Group: GPROF
  time: shape (1,)
  availability_flag: shape (1,)
Group: infrared
  R5_availability_flag: shape (1,)
  R5_time: shape (101,)
  R5: shape (101,)
  infrared_availability_flag: shape (1,)
  time: shape (1,)
  storm_latitude: shape (1,)
  storm_longitude: shape (1,)
  latitude: shape (480, 640)
  longitude: shape (480, 640)
  x: shape (480, 640)
  y: shape (480, 640)
  radius: shape (30,)
  brightness_temperature_thresholds: shape (6,)
  IRWIN: shape (480, 640)
  percent_pixel_below_threshold: shape (6, 1)
  brightness_temperature_annulus_mean: shape (2,)
  brightness_temperature_annulus_standard_deviation: shape (2,)
  brightness_temperature_eye_maximum: shape (1,)
  brightness_temperature_eye_mean: shape (1,)
  brightness_temperature_mean_radial_profile: shape (30,)
  brightness_temperature_radial_profile_eye_maximum_radius: shape (1,)
  brightness_temperature_radial_profile_annulus_minimum: shape (1,)
  brightness_temperature_radial_profile_annulus_mean: shape (1,)
  brightness_temperature_radial_profile_annulus_minimum_radius: shape (1,)
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

file_path = "TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc"
output_dir = "tcprimed_pngs"
os.makedirs(output_dir, exist_ok=True)

# 1. TCPRIMED stores microwave channel data in sub-groups (e.g. passive_microwave/S1 or passive_microwave/S2)
# Let's target "passive_microwave/S1" as the primary scanning group.
group_path = "passive_microwave/S1"
try:
    ds_pmw = xr.open_dataset(file_path, group=group_path)
except Exception:
    # Fallback to general passive_microwave if S1 doesn't exist
    try:
        ds_pmw = xr.open_dataset(file_path, group="passive_microwave")
    except Exception:
        ds_pmw = xr.open_dataset(file_path)

print("Loaded Dataset Structure:")
print(ds_pmw)

# 2. Identify brightness temperature variable (typically 'TB')
tb_var_name = None
for candidate in ["TB", "brightness_temperature", "tb", "precipitation"]:
    if candidate in ds_pmw:
        tb_var_name = candidate
        break

if not tb_var_name:
    # Pick the first 2D or 3D float variable available
    for v in ds_pmw.data_vars:
        if len(ds_pmw[v].shape) >= 2:
            tb_var_name = v
            break

if not tb_var_name:
    raise ValueError("Could not automatically locate a brightness temperature or 2D/3D variable in this group.")

data_array = ds_pmw[tb_var_name]
print(f"\nVisualizing variable: '{tb_var_name}' with shape {data_array.shape}")

# 3. Plot and save each channel / slice
if len(data_array.shape) == 3:
    # Shape is typically (channel, y, x) or (y, x, channel)
    num_channels = data_array.shape[0] if data_array.shape[0] < data_array.shape[-1] else data_array.shape[-1]
    
    for i in range(num_channels):
        slice_2d = data_array[i, :, :] if data_array.shape[0] == num_channels else data_array[:, :, i]
        
        plt.figure(figsize=(7, 6), dpi=150)
        im = plt.imshow(slice_2d, cmap="turbo", origin="lower")
        plt.colorbar(im, label=f"{tb_var_name} ({getattr(data_array, 'units', 'value')})")
        plt.title(f"TCPRIMED - {tb_var_name} (Channel {i+1})")
        plt.xlabel("Cross-track / X")
        plt.ylabel("Along-track / Y")
        plt.tight_layout()
        
        # Save PNG
        out_name = os.path.join(output_dir, f"{tb_var_name}_channel_{i+1}.png")
        plt.savefig(out_name, dpi=200)
        plt.show()
        print(f"Saved: {out_name}")

elif len(data_array.shape) == 2:
    plt.figure(figsize=(7, 6), dpi=150)
    im = plt.imshow(data_array, cmap="turbo", origin="lower")
    plt.colorbar(im, label=f"{tb_var_name} ({getattr(data_array, 'units', 'value')})")
    plt.title(f"TCPRIMED - {tb_var_name}")
    plt.tight_layout()
    
    out_name = os.path.join(output_dir, f"{tb_var_name}.png")
    plt.savefig(out_name, dpi=200)
    plt.show()
    print(f"Saved: {out_name}")   Loaded Dataset Structure: <xarray.Dataset> Size: 200kB Dimensions:       (scan: 65, pixel: 64) Coordinates:     latitude      (scan, pixel) float32 17kB ...     longitude     (scan, pixel) float32 17kB ... Dimensions without coordinates: scan, pixel Data variables:     x             (scan, pixel) float64 33kB ...     y             (scan, pixel) float64 33kB ...     angle_bins    (pixel) int16 128B ...     ScanTime      (scan) datetime64[ns] 520B ...     surface_type  (scan, pixel) float32 17kB ...     TB_19.35V     (scan, pixel) float32 17kB ...     TB_19.35H     (scan, pixel) float32 17kB ...     TB_22.235V    (scan, pixel) float32 17kB ...     TB_37.0V      (scan, pixel) float32 17kB ...     TB_37.0H      (scan, pixel) float32 17kB ... Attributes: (12/50)     title:                                                           NASA int...     summary:                                                         Tropical...     comment:                                                         NASA int...     Conventions:                                                     CF-1.11,...     id:                                                              TCPRIMED...     source:                                                          1C.F11.S...     ...                                                              ...     geospatial_lon_units:                                            degrees_...     time_coverage_start:                                             1999-08-...     time_coverage_end:                                               1999-08-...     automated_tropical_cyclone_forecasting_system_storm_identifier:  al041999     uuid:                                                            e79d5ec2...     date_modified:                                                   2024-11-...  Visualizing variable: 'x' with shape (65, 64) 
image
Saved: tcprimed_pngs/x.png      # Extract latitude, longitude
lat = ds_pmw["latitude"].values
lon = ds_pmw["longitude"].values

# Identify all variables that represent actual brightness temperatures (starting with 'TB_')
tb_variables = [var for var in ds_pmw.data_vars if var.startswith("TB_")]

print(f"Found brightness temperature variables: {tb_variables}")

# Let's plot the first available brightness temperature channel geographically
if tb_variables:
    target_var = tb_variables[0]
    channel_data = ds_pmw[target_var].values

    plt.figure(figsize=(8, 7), dpi=150)
    mesh = plt.pcolormesh(lon, lat, channel_data, cmap="jet", shading="auto")
    plt.colorbar(mesh, label=f"{target_var} Brightness Temperature (K)")
    plt.title(f"TCPRIMED SSM/I - {target_var}")
    plt.xlabel("Longitude (°E)")
    plt.ylabel("Latitude (°N)")
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig("tcprimed_geo_map.png", dpi=200, bbox_inches="tight")
    plt.show()
else:
    print("No TB_ variables found in the dataset.")   Found brightness temperature variables: ['TB_19.35V', 'TB_19.35H', 'TB_22.235V', 'TB_37.0V', 'TB_37.0H']
see this is output of one example file downloaded and i need you take a look and also tell me some code to see all the versions
…
Excellent. This is a useful first inspection of a real TC PRIMED overpass file. We have confirmed that your sample contains passive microwave brightness-temperature data from the SSM/I F11 instrument, alongside storm metadata and infrared-related information.

There are a few important findings and one issue in your plotting code.

1. What your output tells us
Hurricane Beryl | 935mb165mph peak|  postropical - Atlantic Basin - WXSPHERE

Microwave channels

Your file has five brightness-temperature channels:

TB_19.35V

TB_19.35H

TB_22.235V

TB_37.0V

TB_37.0H

Each has a spatial shape of (65, 64) in the S1 group.

Storm metadata

The file also contains storm position, intensity, pressure, development level, and intensity-change information. These are valuable labels for our prediction tasks.

Infrared data

The infrared group contains IRWIN with shape (480, 640), plus latitude/longitude grids and several derived storm statistics. This is another potentially useful input modality.

Important correction about your second plot: it shows x, not a microwave brightness-temperature channel. Your automatic fallback selected the first two-dimensional variable, which happened to be x. That's why you see a smooth coordinate gradient rather than the storm's microwave structure.

Your geographical brightness-temperature plot is the meaningful one. Also, longitude values around 300–317°E are valid; they correspond to approximately 60–43°W.

2. Correct code: plot every microwave channel
Use this instead of automatically selecting the first 2D variable. It explicitly selects every TB_ channel and plots it using the actual geographical coordinates.

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

file_path = "TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc"

ds = xr.open_dataset(
    file_path,
    group="passive_microwave/S1"
)

lat = ds["latitude"].values
lon = ds["longitude"].values

tb_variables = sorted(
    name for name in ds.data_vars
    if name.startswith("TB_")
)

print("Microwave channels:", tb_variables)

fig, axes = plt.subplots(
    2, 3, figsize=(17, 10),
    constrained_layout=True
)

for ax, var_name in zip(axes.flat, tb_variables):
    tb = ds[var_name].values

    # Mask invalid values
    tb = np.ma.masked_invalid(tb)

    mesh = ax.pcolormesh(
        lon, lat, tb,
        shading="auto",
        cmap="turbo"
    )

    ax.set_title(var_name)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.grid(alpha=0.3)

    fig.colorbar(mesh, ax=ax, label="Brightness Temperature (K)")

# Hide the unused sixth subplot
for ax in axes.flat[len(tb_variables):]:
    ax.set_visible(False)

plt.suptitle("TC PRIMED – SSM/I Microwave Channels")
plt.show()

ds.close()

This will let us compare how the same storm appears at different microwave frequencies and polarizations.

3. Inspect every group, subgroup and variable in the NetCDF file
Your current inspection only prints the immediate children of each root group. But passive microwave data has nested groups such as passive_microwave/S1.

Use this recursive inspector to discover the complete structure:

import netCDF4 as nc

file_path = "TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc"

def inspect_group(group, level=0):
    indent = " " * level

    for name, dim in group.dimensions.items():
        print(f"{indent}DIMENSION: {name} = {len(dim)}")

    for name, var in group.variables.items():
        print(f"\n{indent}VARIABLE: {name}")
        print(f"{indent}  Shape: {var.shape}")
        print(f"{indent}  Dimensions: {var.dimensions}")
        print(f"{indent}  Data type: {var.dtype}")

        for attr in var.ncattrs():
            print(f"{indent}  {attr}: {var.getncattr(attr)}")

    for name, subgroup in group.groups.items():
        print(f"\n{indent}========== GROUP: {name} ==========")
        inspect_group(subgroup, level + 4)


with nc.Dataset(file_path, "r") as root:
    print("ROOT ATTRIBUTES")

    for attr in root.ncattrs():
        print(attr, ":", root.getncattr(attr))

    print("\nCOMPLETE NETCDF STRUCTURE")
    inspect_group(root)

This is particularly important because we need to discover the exact dimensions, units, missing-value conventions and coordinate systems before building our data pipeline.

4. Find all TC PRIMED dataset versions in your downloaded directory
If by "all versions" you mean different TC PRIMED releases such as v01r01, v01r02, etc., use this code.

Change data_dir to your downloaded dataset's parent directory.

from pathlib import Path
from collections import Counter
import re

data_dir = Path(r"D:\TC_PRIMED")

version_counts = Counter()
all_files = []

for file in data_dir.rglob("*.nc"):
    match = re.search(r"TCPRIMED_(v\d+r\d+)", file.name)

    if match:
        version = match.group(1)
        version_counts[version] += 1
        all_files.append((version, file))

print("AVAILABLE TC PRIMED VERSIONS")
print("--------------------------------")

for version, count in sorted(version_counts.items()):
    print(f"{version}: {count} files")

print("\nTOTAL NETCDF FILES:", len(all_files))

You can also inspect the version metadata embedded in an individual file:

import netCDF4 as nc

with nc.Dataset(file_path, "r") as ds:
    for attr in ds.ncattrs():
        if any(word in attr.lower() for word in
               ["version", "title", "source", "history", "id"]):
            print(attr, ":", ds.getncattr(attr))

Note: this discovers versions present in your downloaded files; it does not discover releases you haven't downloaded.

5. Our current Phase 1 findings
Component

What we have confirmed

Dataset

TC PRIMED v01r01

Example storm

AL041999

Instrument

SSM/I F11

Microwave channels

5 brightness-temperature variables

Microwave grid

65 × 64

Infrared grid

480 × 640

Geographic coordinates

Available for microwave and IR

Storm metadata

Available

Intensity-change labels

Available, with 9 entries in this example

One important detail: the five microwave channels are not five independent sensors. They are different frequency/polarization measurements from the same instrument.

Also, we should not assume that every storm file has the same instrument, channels, dimensions or available modalities. That is exactly why we are doing this inspection before designing the model.

6. What I want us to inspect next
Let's stay in Phase 1 rather than jumping into model development.

Please run the recursive inspector and share its output. In particular, I want to examine:

passive_microwave/S1 and any other microwave subgroups.

infrared, especially IRWIN, its units and missing values.

The overpass_storm_metadata variables, including intensity and intensity-change units.

Whether this particular file has usable GPROF and radar/radiometer observations.

Our next milestone: build a dataset inventory that tells us exactly which modalities, channels, labels and instruments are available across the downloaded TC PRIMED collection. That inventory will determine the actual multimodal architecture we can responsibly build.

User attachment
ROOT ATTRIBUTES title : Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED) summary : The Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED) is a dataset of satellite passive microwave observations of global tropical cyclones from the NASA GPM Constellation. The TC PRIMED overpass file contains storm metadata interpolated to overpass time, inter-calibrated passive microwave brightness temperatures, NASA GPROF precipitation retrievals, nearly-coincident infrared brightness temperatures from a geostationary satellite (when available), and precipitation variables from the combined radar-radiometer product for the TRMM or GPM satellites (when available). The TC PRIMED environmental file contains storm metadata, environmental diagnostics derived from the ECMWF reanalysis, and gridded rectilinear and re-gridded cylindrical fields from the ECMWF reanalysis at each synoptic time throughout the storm's life. Also included in the environmental file is a list of all overpasses associated with a particular storm. comment : TC PRIMED uses hierarchical groups within its netCDF datasets, which may not be initially visible from some software. Conventions : CF-1.11, ACDD-1.3 id : TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc references : Razin, Muhammad Naufal; Slocum, Christopher J.; Knaff, John A.; Brown, Paula J. 2023. Tropical Cyclone PRecipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED). v01r01. NOAA National Centers for Environmental Information. .  Razin, M. N., C. J. Slocum, J. A. Knaff, P. J. Brown, and M. M. Bell, 2023: Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset (TC PRIMED). Bull. Amer. Meteor. Soc., 104, E1980–E1998, .  Slocum, C. J., M. N. Razin, J. A. Knaff, and J. P. Stow, 2022: Does ERA5 mark a new era for resolving the tropical cyclone environment? J. Climate, 35, 3547-3564.  naming_authority : edu.colostate.cira license : No constraints on data access or use unless otherwise stated date_issued : 2024-07-01T18:00:00.000Z date_metadata_modified : 2024-11-19T02:20:54.000Z product_version : v01r01-final metadata_link :  infoURL :  date_created : 2024-08-18T12:29:51.822Z creator_name : COLOSTATE/CIRA/TC PRIMED TEAM > TC PRIMED TEAM, Cooperative Institute for Research in the Atmosphere, Colorado State University creator_institution : COLOSTATE/CIRA > Cooperative Institute for Research in the Atmosphere, Colorado State University creator_email :  creator_url :  creator_type : group creator_country : USA creator_sector : academic institution : COLOSTATE/CIRA > Cooperative Institute for Research in the Atmosphere, Colorado State University project : TC PRIMED > Tropical Cyclone Precipitation, Infrared, Microwave, and Environmental Dataset publisher_name : DOC/NOAA/NESDIS/NCEI > National Centers for Environmental Information, NESDIS, NOAA, U.S. Department of Commerce publisher_institution : DOC/NOAA/NESDIS/NCEI > National Centers for Environmental Information, NESDIS, NOAA, U.S. Department of Commerce publisher_email :  publisher_url :  publisher_type : institution publisher_country : USA publisher_sector : gov_federal contributor_name : COLOSTATE/ATMOS > Department of Atmospheric Science, Colorado State University  DOC/NOAA/NESDIS/STAR > Center for Satellite Applications and Research, NESDIS, NOAA, U.S. Department of Commerce keywords : EARTH SCIENCE SERVICES > DATA MANAGEMENT/DATA HANDLING > SUBSETTING/SUPERSETTING, EARTH SCIENCE > ATMOSPHERE > WEATHER EVENTS > TROPICAL CYCLONES, PLATFORMS > SPACE-BASED PLATFORMS > EARTH OBSERVATION SATELLITES keywords_vocabulary : GCMD:Global Change Master Directory Keywords v18.7 standard_name_vocabulary : CF Standard Name Table v85 automated_tropical_cyclone_forecasting_system_storm_identifier : al041999 uuid : e79d5ec2-a61c-11ef-b814-3cecefd0da00 data_groups : overpass_metadata overpass_storm_metadata passive_microwave infrared history : Created 2024-08-18T12:29:51.822Z Modified 2024-11-19T02:20:54.000Z to correct overpass_storm_metadata/storm_heading date_modified : 2024-11-19T02:20:54.000Z  COMPLETE NETCDF STRUCTURE  ========== GROUP: overpass_metadata ==========     DIMENSION: time = 1      VARIABLE: time       Shape: (1,)       Dimensions: ('time',)       Data type: int64       long_name: Time of the passive microwave observation of the storm       standard_name: time       axis: T       units: seconds since 1970-01-01T00:00:00Z       calendar: standard       coverage_content_type: coordinate      VARIABLE: basin       Shape: (1,)       Dimensions: ('time',)       Data type: <class 'str'>       long_name: Tropical cyclone basin       comment: AL - North Atlantic basin, north of the Equator; SL - South Atlantic basin, south of the Equator; EP - North East Pacific basin, eastward of 140 degrees west longitude; CP - North Central Pacific basin, between the dateline and 140 degrees west longitude; WP - North West Pacific basin, westward of the dateline; IO - North Indian Ocean basin, north of the Equator between 40 and 100 degrees east longitude; SH - South Pacific Ocean basin and South Indian Ocean basin.       coverage_content_type: thematicClassification      VARIABLE: cyclone_number       Shape: (1,)       Dimensions: ('time',)       Data type: int8       long_name: Tropical cyclone annual cyclone number       comment: Assigned once a system develops into a tropical or subtropical cyclone       units: 1       valid_range: [ 1 49]       coverage_content_type: thematicClassification      VARIABLE: season       Shape: (1,)       Dimensions: ('time',)       Data type: int16       long_name: Tropical cyclone basin season       comment: The calendar year for the Northern Hemisphere. For the Southern Hemisphere, the year begins July 1, with calendar year plus one.       coverage_content_type: thematicClassification      VARIABLE: instrument_name       Shape: (1,)       Dimensions: ('time',)       Data type: <class 'str'>       long_name: Satellite instrument name       coverage_content_type: auxiliaryInformation      VARIABLE: platform_name       Shape: (1,)       Dimensions: ('time',)       Data type: <class 'str'>       long_name: Satellite platform name       standard_name: platform_id       coverage_content_type: auxiliaryInformation      VARIABLE: granule_number       Shape: (1,)       Dimensions: ('time',)       Data type: <class 'str'>       long_name: Satellite granule number for the tropical cyclone overpass       comment: A satellite granule begins and ends at the southernmost point of the satellite ground track during its orbit.       coverage_content_type: auxiliaryInformation      VARIABLE: coverage_fraction       Shape: (1,)       Dimensions: ('time',)       Data type: int16       long_name: Amount of satellite passive microwave coverage from the S1 swath within 750 km radius centered on the storm center       standard_name: area_fraction       units: %       scale_factor: 0.01       valid_range: [    0 10000]       actual_range: [  0. 100.]       coverage_content_type: qualityInformation      VARIABLE: env_time       Shape: (1,)       Dimensions: ('time',)       Data type: int64       long_name: Closest standard synoptic time (0000, 0600, 1200, 1800 UTC) to the passive microwave observation time for TC PRIMED environmental file       standard_name: time       units: seconds since 1970-01-01T00:00:00Z       calendar: standard       coverage_content_type: referenceInformation  ========== GROUP: overpass_storm_metadata ==========     DIMENSION: time = 1     DIMENSION: intensity_change_periods = 9      VARIABLE: time       Shape: (1,)       Dimensions: ('time',)       Data type: int64       long_name: Time of the passive microwave observation of the storm       standard_name: time       axis: T       units: seconds since 1970-01-01T00:00:00Z       calendar: standard       coverage_content_type: coordinate      VARIABLE: storm_latitude       Shape: (1,)       Dimensions: ('time',)       Data type: float32       long_name: Tropical cyclone center latitude interpolated from the best-track to the passive microwave observation time       standard_name: latitude       units: degrees_north       valid_range: [-90.  90.]       interpolation_description: cubic b-spline       coverage_content_type: coordinate      VARIABLE: storm_longitude       Shape: (1,)       Dimensions: ('time',)       Data type: float32       long_name: Tropical cyclone center longitude interpolated from the best-track to the passive microwave observation time       standard_name: longitude       units: degrees_east       valid_range: [  0. 360.]       interpolation_description: cubic b-spline       coverage_content_type: coordinate      VARIABLE: intensity       Shape: (1,)       Dimensions: ('time',)       Data type: int16       long_name: Tropical cyclone intensity linearly interpolated from the best-track to the passive microwave observation time; maximum 1-min average sustained wind speed at 10 m       standard_name: tropical_cyclone_maximum_sustained_wind_speed       units: kt       valid_range: [  0 300]       _FillValue: -999       missing_value: -999       interpolation_description: linear       coverage_content_type: physicalMeasurement      VARIABLE: intensity_change_periods       Shape: (9,)       Dimensions: ('intensity_change_periods',)       Data type: int8       long_name: Periods of intensity change relative to the passive microwave observation time       units: hours       valid_range: [-24  24]       comment: Negative values indicate past i

…
Your inspection confirms that TC PRIMED contains multiple observation types inside the same NetCDF overpass file. We should therefore build one reusable notebook that detects the available groups and channels automatically, rather than writing separate code for every satellite or instrument.

For your three sample files:

AL041999_SSMI_F11... — SSM/I microwave observations, with the channel groups shown in your output.

EP071999_SSMI_F13... — another SSM/I overpass; its available channels and dimensions may differ.

IO031999_TMI_TRMM... — a TMI/TRMM overpass; it may have a different channel configuration.

The code below will handle these differences by inspecting each file's actual structure. It will cover:

Infrared (IR): IRWIN, latitude and longitude.

Passive microwave: every available TB_* channel in S1, S2, or other subgroups.

GPROF precipitation: surface precipitation, convective precipitation, water paths and other available fields.

Radar-radiometer: fields when that group exists.

Metadata and labels: storm position, wind speed, pressure, development level, basin and observation time.

Batch inspection: run the same analysis on multiple .nc files.

We'll preserve the original scientific values in Kelvin, mm/hr, etc. The plots are only for visualization.

1. Install and import the libraries
Run this first in your Colab notebook:

!pip install netCDF4 numpy matplotlib

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset, num2date

2. Build a universal NetCDF structure inspector
This is important because SSM/I, TMI and other instruments may not have identical groups or channels.

def walk_groups(group, prefix=""):
    """
    Recursively visits every group and subgroup in a NetCDF file.
    """
    for name, subgroup in group.groups.items():
        path = f"{prefix}/{name}" if prefix else name

        yield path, subgroup

        yield from walk_groups(subgroup, path)


def inspect_netcdf(file_path):
    """
    Prints the structure of a TC PRIMED NetCDF file.
    """

    print("=" * 100)
    print("FILE:", os.path.basename(file_path))
    print("=" * 100)

    with Dataset(file_path, "r") as nc:

        print("\nROOT ATTRIBUTES")

        for attr in nc.ncattrs():
            print(f"{attr}: {getattr(nc, attr)}")

        print("\nGROUP STRUCTURE")

        for group_path, group in walk_groups(nc):

            print(f"\n{'-' * 80}")
            print("GROUP:", group_path)

            print("\nDimensions:")

            for name, dimension in group.dimensions.items():
                print(f"  {name}: {len(dimension)}")

            print("\nVariables:")

            for name, variable in group.variables.items():

                print(f"\n  Variable: {name}")
                print("    Shape:", variable.shape)
                print("    Dimensions:", variable.dimensions)
                print("    Data type:", variable.dtype)

                for attr in ["long_name", "units", "frequency",
                             "polarization", "valid_range"]:

                    if attr in variable.ncattrs():
                        print(
                            f"    {attr}: "
                            f"{getattr(variable, attr)}"
                        )

        print("\nINSPECTION COMPLETE")

Run it on your three sample files:

files = [
    "/content/TCPRIMED_v01r01-final_AL041999_SSMI_F11_039954_19990825224321.nc",

    "/content/TCPRIMED_v01r01-final_EP071999_SSMI_F13_022614_19990812020754.nc",

    "/content/TCPRIMED_v01r01-final_IO031999_TMI_TRMM_008844_19990611171415.nc"
]

for file_path in files:

    if os.path.exists(file_path):
        inspect_netcdf(file_path)

    else:
        print("FILE NOT FOUND:", file_path)

This will tell us exactly which channels and observation groups exist in each file. It won't assume that every file has S1, S2, GPROF or infrared.

3. Create reusable functions for reading and plotting satellite data
The next functions will automatically locate latitude, longitude and the actual data variable.

def get_group(nc, group_path):


def get_array(group, variable_name):
    """
    Reads a variable while preserving missing-value masks.
    """

    if variable_name not in group.variables:
        return None

    variable = group.variables[variable_name]

    data = variable[:]

    if np.ma.isMaskedArray(data):
        data = np.ma.masked_invalid(data)
    else:
        data = np.asarray(data, dtype=np.float32)
        data = np.ma.masked_invalid(data)

    return data


def plot_geospatial_field(
    latitude,
    longitude,
    data,
    title,
    units="",
    cmap="turbo"
):
    """
    Plots a satellite field using its actual geographic coordinates.
    """

    latitude = np.ma.masked_invalid(latitude)
    longitude = np.ma.masked_invalid(longitude)
    data = np.ma.masked_invalid(data)

    if latitude.shape != longitude.shape:
        print("Latitude/longitude shape mismatch.")
        return

    if data.shape != latitude.shape:
        print(
            "Data and coordinate shape mismatch:",
            data.shape,
            latitude.shape
        )
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.pcolormesh(
        longitude,
        latitude,
        data,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(
        image,
        ax=ax,
        label=units
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")

    ax.grid(alpha=0.3)


Important: TC PRIMED microwave and GPROF data are swath observations, not regular rectangular latitude-longitude grids. pcolormesh() uses the provided geographic coordinates, which is why we use the actual latitude and longitude arrays instead of assuming a regular grid.

4. Infrared visualization
Your example has:

Group: infrared

Image variable: IRWIN

Latitude: latitude

Longitude: longitude

Units: Kelvin

Here is a function that checks whether the IR data exists before plotting it.

def plot_infrared(file_path):

    with Dataset(file_path, "r") as nc:

        if "infrared" not in nc.groups:
            print("Infrared group is not available.")
            return

        group = nc.groups["infrared"]

        # Check availability flag
        if "infrared_availability_flag" in group.variables:

            flag = group.variables[
                "infrared_availability_flag"
            ][0]

            print("IR availability flag:", flag)

            if flag == 0:
                print("Infrared observation is unavailable.")
                return

        required = ["IRWIN", "latitude", "longitude"]

        for name in required:

            if name not in group.variables:
                print("Missing IR variable:", name)
                return

        ir = get_array(group, "IRWIN")
        latitude = get_array(group, "latitude")
        longitude = get_array(group, "longitude")

        plot_geospatial_field(
            latitude,
            longitude,
            ir,
            title="TC PRIMED - Infrared Brightness Temperature",
            units="Brightness Temperature (K)",
            cmap="turbo"
        )

Example:

plot_infrared(files[0])
plot_infrared(files[1])
plot_infrared(files[2])

This will work when the respective file has an available infrared observation.

5. Passive microwave visualization
This is particularly important for our project.

Your example includes:

Channel

Frequency

Polarization

TB_19.35H

19.35 GHz

Horizontal

TB_19.35V

19.35 GHz

Vertical

TB_22.235V

22.235 GHz

Vertical

TB_37.0H

37.0 GHz

Horizontal

TB_37.0V

37.0 GHz

Vertical

TB_85.5H

85.5 GHz

Horizontal

TB_85.5V

85.5 GHz

Vertical

These are the channels visible in your supplied file inspection. Other instruments may expose different channels.

The following code discovers every available TB_* variable automatically.

def plot_passive_microwave(file_path):

    with Dataset(file_path, "r") as nc:

        if "passive_microwave" not in nc.groups:
            print("Passive microwave group unavailable.")
            return

        microwave = nc.groups["passive_microwave"]

        channel_count = 0

        for group_path, group in walk_groups(microwave, "passive_microwave"):

            if "latitude" not in group.variables:
                continue

            if "longitude" not in group.variables:
                continue

            latitude = get_array(group, "latitude")
            longitude = get_array(group, "longitude")

            for variable_name, variable in group.variables.items():

                if not variable_name.startswith("TB_"):
                    continue

                data = get_array(group, variable_name)

                if data is None:
                    continue

                units = getattr(variable, "units", "")
                frequency = getattr(variable, "frequency", "")
                polarization = getattr(variable, "polarization", "")

                title = (
                    f"{variable_name} | "
                    f"{frequency} | "
                    f"{polarization}\n"
                    f"{group_path}"
                )

                plot_geospatial_field(
                    latitude,
                    longitude,
                    data,
                    title=title,
                    units=f"Brightness Temperature ({units})",
                    cmap="turbo"
                )

                channel_count += 1

        if channel_count == 0:
            print("No TB_* channels found.")

        else:
            print("Total microwave channels plotted:", channel_count)

Run:

for file_path in files:

    print("\nPROCESSING:", os.path.basename(file_path))

    plot_passive_microwave(file_path)

This function does not assume that every instrument has the same channels or subgroup names. It discovers the available TB_* variables inside the microwave hierarchy.

6. GPROF precipitation visualization
GPROF gives us additional information beyond brightness temperature.

Useful fields include:

surfacePrecipitation

convectivePrecipitation

rainWaterPath

cloudWaterPath

iceWaterPath

probabilityOfPrecip

qualityFlag

Some fields are vertically resolved, such as rainWaterContent, cloudWaterContent and snowWaterContent.

We'll begin with the two-dimensional fields.

GPROF_FIELDS = [
    "surfacePrecipitation",
    "convectivePrecipitation",
    "rainWaterPath",
    "cloudWaterPath",
    "iceWaterPath",
    "probabilityOfPrecip",
    "qualityFlag"
]


def plot_gprof(file_path):

    with Dataset(file_path, "r") as nc:

        if "GPROF" not in nc.groups:
            print("GPROF group unavailable.")
            return

        gprof = nc.groups["GPROF"]

        if "availability_flag" in gprof.variables:

            flag = gprof.variables["availability_flag"][0]

            print("GPROF availability flag:", flag)

            if flag == 0:
                print("GPROF data unavailable.")
                return

        count = 0

        for group_path, group in walk_groups(gprof, "GPROF"):

            if "latitude" not in group.variables:

Run:

for file_path in files:

    print("\nGPROF:", os.path.basename(file_path))

    plot_gprof(file_path)

A note about GPROF quality flags
qualityFlag is not a precipitation measurement. It indicates retrieval quality, with the meanings shown in your metadata:

0: good

1: use with caution

2: use with extreme care over snow-covered surfaces

3: use with extreme caution

We should use these flags during preprocessing rather than treating them as ordinary input channels without consideration.

7. Radar-radiometer visualization
Some overpass files contain a radar_radiometer group. It may not exist in every file, so this function checks automatically.

def plot_radar_radiometer(file_path):

    with Dataset(file_path, "r") as nc:

        if "radar_radiometer" not in nc.groups:
            print("Radar-radiometer data unavailable.")
            return

        radar = nc.groups["radar_radiometer"]

        count = 0

        for group_path, group in walk_groups(
            radar, "radar_radiometer"
        ):

            if "latitude" not in group.variables:
                continue

            if "longitude" not in group.variables:
                continue

            latitude = get_array(group, "latitude")
            longitude = get_array(group, "longitude")

            excluded = {
                "latitude", "longitude", "x", "y",
                "time", "ScanTime", "angle_bins"
            }

            for name, variable in group.variables.items():

                if name in excluded:
                    continue

                if len(variable.dimensions) != 2:
                    continue

                data = get_array(group, name)

                if data is None:
                    continue

                # Only plot fields matching the geographic grid.
                if data.shape != latitude.shape:
                    continue

                units = getattr(variable, "units", "")

                plot_geospatial_field(
                    latitude,
                    longitude,
                    data,
                    title=f"Radar-Radiometer - {name}",
                    units=units,
                    cmap="turbo"
                )


8. Extract storm metadata and prediction labels
This part is especially important because these variables will eventually become our training labels.

From your sample, we have:

Variable

Meaning

Unit

storm_latitude

Cyclone centre latitude

Degrees

storm_longitude

Cyclone centre longitude

Degrees

intensity

Maximum sustained wind

kt

central_min_pressure

Minimum central pressure

hPa

development_level

Cyclone development category

Category

intensity_change

Intensity change at different periods

kt

storm_speed

Translational speed

m/s

storm_heading

Direction of movement

Degrees

Let's extract them without assuming every label is present.

def print_storm_metadata(file_path):

    with Dataset(file_path, "r") as nc:

        print("\n" + "=" * 80)
        print("STORM METADATA")
        print(os.path.basename(file_path))
        print("=" * 80)

        for group_name in [
            "overpass_metadata",
            "overpass_storm_metadata"
        ]:

            if group_name not in nc.groups:
                continue

            group = nc.groups[group_name]

            print(f"\nGROUP: {group_name}")

            selected_variables = [
                "basin",
                "season",
                "cyclone_number",
                "instrument_name",
                "platform_name",
                "time",
                "storm_latitude",
                "storm_longitude",
                "intensity",
                "central_min_pressure",
                "development_level",
                "storm_speed",
                "storm_heading",
                "distance_to_land",

Run:

for file_path in files:

    if os.path.exists(file_path):
        print_storm_metadata(file_path)

One important detail: intensity_change_periods contains the time offsets, while intensity_change contains the corresponding changes in wind speed. We must match these arrays by index when building our training labels.

For example, the future 24-hour intensity-change label should be selected using the entry whose period is 24, not by assuming a fixed array index.

9. Inspect vertically resolved precipitation profiles
GPROF also contains three-dimensional atmospheric fields, for example:

rainWaterContent

cloudWaterContent

snowWaterContent

These have dimensions such as:

(hgtTopLayer, scan, pixel)
We can visualize a horizontal slice at a selected atmospheric layer.

def plot_gprof_vertical_profile(
    file_path,
    variable_name="rainWaterContent",
    layer_index=10
):

    with Dataset(file_path, "r") as nc:

        if "GPROF" not in nc.groups:
            print("GPROF unavailable.")
            return

        gprof = nc.groups["GPROF"]

        for group_path, group in walk_groups(gprof, "GPROF"):

            if variable_name not in group.variables:
                continue

            variable = group.variables[variable_name]

            if len(variable.dimensions) != 3:
                continue

            data = get_array(group, variable_name)

            if not 0 <= layer_index < data.shape[0]:
                print("Invalid layer index.")
                return

            latitude = get_array(group, "latitude")
            longitude = get_array(group, "longitude")

            heights = get_array(group, "hgtTopLayer")

            layer_data = data[layer_index, :, :]

Example:

plot_gprof_vertical_profile(
    files[0],
    variable_name="rainWaterContent",
    layer_index=10
)

This gives us a view of precipitation content at one altitude. It is not a complete 3D reconstruction.

10. One function to run all available observations
Instead of calling every function individually, use this:

def analyze_tc_primed_file(file_path):

    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    print("\n" + "#" * 100)
    print("ANALYZING:", os.path.basename(file_path))
    print("#" * 100)

    # 1. Structure
    inspect_netcdf(file_path)

    # 2. Metadata and labels
    print_storm_metadata(file_path)

    # 3. Infrared
    print("\nINFRARED ANALYSIS")
    plot_infrared(file_path)

    # 4. Passive microwave
    print("\nPASSIVE MICROWAVE ANALYSIS")
    plot_passive_microwave(file_path)

    # 5. GPROF
    print("\nGPROF ANALYSIS")
    plot_gprof(file_path)

    # 6. Radar-radiometer
    print("\nRADAR-RADIOMETER ANALYSIS")
    plot_radar_radiometer(file_path)

Run one file first:

analyze_tc_primed_file(files[0])

I recommend doing this on just one file initially because it can generate quite a few plots.

11. Batch-process all your TC PRIMED files
Once the functions work on your sample files, we can inspect an entire directory.

DATA_DIR = "/content"

all_nc_files = glob.glob(
    os.path.join(DATA_DIR, "TCPRIMED*.nc")
)

print("Total NetCDF files found:", len(all_nc_files))

for file_path in all_nc_files:

    print("\nProcessing:", os.path.basename(file_path))

    print_storm_metadata(file_path)

For now, use this batch operation for metadata. Don't generate every image for thousands of files yet.

12. What we should build from this inspection
TC PRIMED data intelligence pipeline
Raw TC PRIMED NetCDF files

SSM/I · TMI · GMI · other available instruments

Infrared

IRWIN · brightness temperature

Microwave

TB channels · frequencies · polarization

GPROF

Rain · cloud · ice · precipitation

Environmental data

ERA5 fields from environmental files

Data catalogue and quality control

Shapes · units · missing values · timestamps · sensor availability

Aligned, storm-centred training samples

Spatial alignment · temporal matching · masks · normalization

Important research decisions before training
There are three things we must not overlook:

Different spatial resolutions: Your example's microwave channels have different footprints. We should not assume that every channel represents the same spatial resolution just because the arrays have matching dimensions.

Missing modalities: Some overpasses have no IR or GPROF data. We should catalogue availability rather than discard all such observations.

No data leakage: All observations belonging to the same cyclone must remain in the same train, validation or test split.

What I want you to run next
Start with these three files only:

for file_path in files:
    inspect_netcdf(file_path)
    print_storm_metadata(file_path)

Then share the outputs for the TMI file (IO031999), particularly its passive_microwave groups and channel names.

After that, we'll create a proper Phase 1 data catalogue that automatically processes the full dataset and produces:

A CSV containing every file, storm ID, basin, instrument and platform.

A channel inventory showing which frequencies and polarizations are available.

An IR/GPROF/microwave availability report.

A variable inventory with dimensions, units and missing-value statistics.

A storm-wise and basin-wise dataset summary.

That catalogue will become the foundation of our preprocessing and model architecture decisions.

