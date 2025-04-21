declare namespace Cloudflare {
	interface Env {
		browser_search: R2Bucket;
		ASSETS: Fetcher;
	}
}
interface Env extends Cloudflare.Env {}
