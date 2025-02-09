// pages/api/news.js (Next.js) or /api/news.js in some serverless environment

export default async function handler(req, res) {
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Method Not Allowed' });
    }
  
    // In a production environment, you'd typically read your key from
    // process.env.NEWS_API_KEY, rather than from the client's request body:
    // const apiKey = process.env.NEWS_API_KEY;
    // For demonstration, we'll read it from req.body here:
    const { apiKey, query, language, sortBy, fromDate, toDate } = req.body;
  
    if (!apiKey) {
      return res.status(400).json({ error: 'API key is required' });
    }
  
    const url = new URL('https://newsapi.org/v2/everything');
    if (query) url.searchParams.append('q', query);
    if (language) url.searchParams.append('language', language);
    if (sortBy) url.searchParams.append('sortBy', sortBy);
    if (fromDate) url.searchParams.append('from', fromDate);
    if (toDate) url.searchParams.append('to', toDate);
    url.searchParams.append('apiKey', apiKey);
  
    try {
      const response = await fetch(url);
      const data = await response.json();
      res.status(200).json(data);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch news', details: error.message });
    }
  }
  