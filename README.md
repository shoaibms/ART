---
generator: Riched20 10.0.19041
---

**Code Organisation and Naming Conventions**  
The code is organised to clearly reflect its functionality, making it
easier for users to navigate and understand. The structure is as
follows:

- **Programming Languages**: The code is first categorised by the
  programming language used.

- **Python Folder**:

  1.  **Algorithms for ART Generation**: Contains scripts for generating
      Algorithmic Root Traits (ART).

  2.  **Classification**: Includes scripts used for classification tasks
      based on the ART and other data.

  3.  **Rest**: Contains additional scripts that support data analysis
      and preprocessing.

<!-- -->

- **R Folder**:  
  Contains scripts for plotting and visualisation.

- **Data Folder**:  
  This folder houses the datasets used in the analysis, with file names
  aligned with those in the manuscript for easy reference and clarity.

**Environment Setup**  
To ensure a smooth setup and avoid potential conflicts, we recommend
using a virtual environment. Follow these steps:

*Create a virtual environment:*

python -m venv env_name

Replace env_name with your preferred name for the virtual environment.

*Activate the virtual environment:*

On Windows:

.\env_name\Scripts\activate

*On macOS/Linux:*

source env_name/bin/activate

*Install the necessary dependencies:*

You can manually install the required packages in this virtual environment based on your needs. 
If there are conflicts, you can create a separate environment to resolve them.
