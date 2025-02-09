export default async function handler(req, res) {
    if (req.method !== 'POST') {
      return res.status(405).json({ error: 'Method Not Allowed' });
    }
  
    const { apiKey } = req.body;
    if (!apiKey) {
      return res.status(400).json({ error: 'API key is required' });
    }
  
    const url = new URL('https://newsapi.org/v2/sources');
    url.searchParams.append('apiKey', apiKey);
  
    try {
      const response = await fetch(url);
      const data = await response.json();
      // The "status" on success is "ok", with an array "sources".
      return res.status(200).json(data);
    } catch (error) {
      res.status(500).json({
        error: 'Failed to fetch sources',
        details: error.message,
      });
    }
  }

  


  