export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		try {
			const url = new URL(request.url);
			const path = url.pathname;

			// Proxy Hugging Face model requests
			if (path.startsWith('/proxy/huggingface/')) {
				const modelPath = path.replace('/proxy/huggingface/', '');
				const huggingFaceUrl = `https://huggingface.co/${modelPath}`;

				console.log('Proxying request to:', huggingFaceUrl); // Debug log

				// Forward the request to Hugging Face
				const response = await fetch(huggingFaceUrl, {
					method: request.method,
					headers: {
						'User-Agent': 'Mozilla/5.0 (compatible; CloudflareWorker/1.0)',
						Accept: request.headers.get('Accept') || '*/*',
						'Cache-Control': 'no-cache',
					},
				});

				// Create a new response with CORS headers
				const corsResponse = new Response(response.body, {
					status: response.status,
					statusText: response.statusText,
					headers: {
						...Object.fromEntries(response.headers),
						'Access-Control-Allow-Origin': '*',
						'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
						'Access-Control-Allow-Headers': 'Content-Type, Authorization',
						'Access-Control-Allow-Credentials': 'false',
					},
				});

				return corsResponse;
			}

			// Handle preflight requests
			if (request.method === 'OPTIONS') {
				return new Response(null, {
					headers: {
						'Access-Control-Allow-Origin': '*',
						'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
						'Access-Control-Allow-Headers': 'Content-Type, Authorization',
						'Access-Control-Max-Age': '86400',
					},
				});
			}

			// Serve static assets
			const asset = await env.ASSETS.fetch(request);
			if (asset.status >= 200 && asset.status < 300) {
				// Add CORS headers to allow cross-origin requests
				const response = new Response(asset.body, asset);
				response.headers.set('Access-Control-Allow-Origin', '*');
				response.headers.set('Content-Type', asset.headers.get('Content-Type') || 'application/octet-stream');
				return response;
			}

			// Fallback for root or invalid paths
			if (path === '/' || path === '') {
				return env.ASSETS.fetch(new Request(`${request.url}index.html`));
			}

			return new Response('Resource not found. Visit /index.html to start.', {
				status: 404,
				headers: { 'Content-Type': 'text/plain' },
			});
		} catch (error) {
			console.error('Worker error:', error);
			return new Response('Internal Server Error', { status: 500 });
		}
	},
} satisfies ExportedHandler<Env>;
