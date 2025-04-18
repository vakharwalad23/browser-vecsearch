export default {
	async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
		try {
			const url = new URL(request.url);
			const path = url.pathname;

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
