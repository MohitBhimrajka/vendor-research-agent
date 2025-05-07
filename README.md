# Supervity Vendor Research Agent

Supervity's Vendor Research Agent is a powerful Streamlit application that uses Gemini to find, research, and recommend real vendors for any product, service, or industry. The application follows a wizard-style interface that guides you through term disambiguation, vendor count & mix selection, and displays results with filtering and sorting capabilities.

## Features

- **Term Disambiguation**: Automatically generate 2-3 interpretations of your search term
- **Smart Batching**: Generate balanced vendor lists with manufacturers, distributors, and retailers
- **Vendor Enrichment**: Automatically generate realistic company profiles including:
  - Detailed descriptions
  - Website URLs
  - Contact information
  - Specializations
  - Relevance scores
  - Business type classification
- **Real-time Processing**: Skeleton loaders and progress indicators during LLM calls
- **Filtering & Sorting**: Filter by score or business type, sort by various criteria
- **Responsive UI**: Clean, modern interface with styled vendor cards

## Architecture

The application is built with a modular architecture:

- **app.py**: Main Streamlit interface with the wizard flow
- **llm_service.py**: LLM prompt construction and API call logic
- **vendor_manager.py**: Vendor generation and enrichment functionality
- **utils.py**: Helper functions for batching, caching, and UI components

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/vendor-research-agent.git
   cd vendor-research-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Gemini API key:
   - Create a `.env` file in the project root
   - Add your API key: `GEMINI_API_KEY=your-api-key-here`

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. Open your browser and go to `http://localhost:8501`

3. Follow the wizard steps:
   - Enter a search term (e.g., "solar panels", "3D printing", "IT consulting")
   - Select the most relevant interpretation
   - Configure vendor count and business type mix
   - View, filter, and sort the generated vendors

## Example

1. Search for "solar panels"
2. Select the interpretation "Solar panel manufacturers and suppliers"
3. Configure for 20 vendors with 40% manufacturers, 30% distributors, 30% retailers
4. View the generated vendor list
5. Filter by relevance score > 7
6. Sort by relevance score from high to low

## Requirements

- Python 3.7+
- Streamlit
- Google Gemini API key

## Performance & Reliability

- Caching to avoid redundant LLM calls
- Retry mechanisms with exponential backoff
- Asynchronous processing for improved performance
- Error handling with fallback strategies

## License

[MIT License](LICENSE) 