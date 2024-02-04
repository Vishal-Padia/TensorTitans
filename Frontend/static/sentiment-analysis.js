// Assuming you have a button with id 'analyze-button' in your HTML
document.getElementById('analyze-button')

  if (!url) {
    alert('Please enter a YouTube URL');
  }

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ url: url }),
    });

    const data = await response.json();

    // Display sentiment result using the data from the server
    displaySentimentResult(data.sentiment);

  } catch (error) {
    console.error('Error:', error);
    alert('Failed to analyze sentiment');
  }
});
