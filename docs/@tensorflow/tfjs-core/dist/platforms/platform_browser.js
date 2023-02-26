/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import '../flags';
import { env } from '../environment';
import { BrowserIndexedDB, BrowserIndexedDBManager } from '../io/indexed_db';
import { BrowserLocalStorage, BrowserLocalStorageManager } from '../io/local_storage';
import { ModelStoreManagerRegistry } from '../io/model_management';
export class PlatformBrowser {
    constructor() {
        // For setTimeoutCustom
        this.messageName = 'setTimeoutCustom';
        this.functionRefs = [];
        this.handledMessageCount = 0;
        this.hasEventListener = false;
    }
    fetch(path, init) {
        return fetch(path, init);
    }
    now() {
        return performance.now();
    }
    encode(text, encoding) {
        if (encoding !== 'utf-8' && encoding !== 'utf8') {
            throw new Error(`Browser's encoder only supports utf-8, but got ${encoding}`);
        }
        if (this.textEncoder == null) {
            this.textEncoder = new TextEncoder();
        }
        return this.textEncoder.encode(text);
    }
    decode(bytes, encoding) {
        return new TextDecoder(encoding).decode(bytes);
    }
    // If the setTimeout nesting level is greater than 5 and timeout is less
    // than 4ms, timeout will be clamped to 4ms, which hurts the perf.
    // Interleaving window.postMessage and setTimeout will trick the browser and
    // avoid the clamp.
    setTimeoutCustom(functionRef, delay) {
        if (typeof window === 'undefined' ||
            !env().getBool('USE_SETTIMEOUTCUSTOM')) {
            setTimeout(functionRef, delay);
            return;
        }
        this.functionRefs.push(functionRef);
        setTimeout(() => {
            window.postMessage({ name: this.messageName, index: this.functionRefs.length - 1 }, '*');
        }, delay);
        if (!this.hasEventListener) {
            this.hasEventListener = true;
            window.addEventListener('message', (event) => {
                if (event.source === window && event.data.name === this.messageName) {
                    event.stopPropagation();
                    const functionRef = this.functionRefs[event.data.index];
                    functionRef();
                    this.handledMessageCount++;
                    if (this.handledMessageCount === this.functionRefs.length) {
                        this.functionRefs = [];
                        this.handledMessageCount = 0;
                    }
                }
            }, true);
        }
    }
    isTypedArray(a) {
        return a instanceof Float32Array || a instanceof Int32Array ||
            a instanceof Uint8Array || a instanceof Uint8ClampedArray;
    }
}
if (env().get('IS_BROWSER')) {
    env().setPlatform('browser', new PlatformBrowser());
    // Register LocalStorage IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserLocalStorage.URL_SCHEME, new BrowserLocalStorageManager());
    }
    catch (err) {
    }
    // Register IndexedDB IOHandler
    try {
        ModelStoreManagerRegistry.registerManager(BrowserIndexedDB.URL_SCHEME, new BrowserIndexedDBManager());
    }
    catch (err) {
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicGxhdGZvcm1fYnJvd3Nlci5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvcGxhdGZvcm1zL3BsYXRmb3JtX2Jyb3dzZXIudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBRUgsT0FBTyxVQUFVLENBQUM7QUFFbEIsT0FBTyxFQUFDLEdBQUcsRUFBQyxNQUFNLGdCQUFnQixDQUFDO0FBQ25DLE9BQU8sRUFBQyxnQkFBZ0IsRUFBRSx1QkFBdUIsRUFBQyxNQUFNLGtCQUFrQixDQUFDO0FBQzNFLE9BQU8sRUFBQyxtQkFBbUIsRUFBRSwwQkFBMEIsRUFBQyxNQUFNLHFCQUFxQixDQUFDO0FBQ3BGLE9BQU8sRUFBQyx5QkFBeUIsRUFBQyxNQUFNLHdCQUF3QixDQUFDO0FBSWpFLE1BQU0sT0FBTyxlQUFlO0lBQTVCO1FBS0UsdUJBQXVCO1FBQ04sZ0JBQVcsR0FBRyxrQkFBa0IsQ0FBQztRQUMxQyxpQkFBWSxHQUFlLEVBQUUsQ0FBQztRQUM5Qix3QkFBbUIsR0FBRyxDQUFDLENBQUM7UUFDeEIscUJBQWdCLEdBQUcsS0FBSyxDQUFDO0lBK0RuQyxDQUFDO0lBN0RDLEtBQUssQ0FBQyxJQUFZLEVBQUUsSUFBa0I7UUFDcEMsT0FBTyxLQUFLLENBQUMsSUFBSSxFQUFFLElBQUksQ0FBQyxDQUFDO0lBQzNCLENBQUM7SUFFRCxHQUFHO1FBQ0QsT0FBTyxXQUFXLENBQUMsR0FBRyxFQUFFLENBQUM7SUFDM0IsQ0FBQztJQUVELE1BQU0sQ0FBQyxJQUFZLEVBQUUsUUFBZ0I7UUFDbkMsSUFBSSxRQUFRLEtBQUssT0FBTyxJQUFJLFFBQVEsS0FBSyxNQUFNLEVBQUU7WUFDL0MsTUFBTSxJQUFJLEtBQUssQ0FDWCxrREFBa0QsUUFBUSxFQUFFLENBQUMsQ0FBQztTQUNuRTtRQUNELElBQUksSUFBSSxDQUFDLFdBQVcsSUFBSSxJQUFJLEVBQUU7WUFDNUIsSUFB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OlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTkgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQgJy4uL2ZsYWdzJztcblxuaW1wb3J0IHtlbnZ9IGZyb20gJy4uL2Vudmlyb25tZW50JztcbmltcG9ydCB7QnJvd3NlckluZGV4ZWREQiwgQnJvd3NlckluZGV4ZWREQk1hbmFnZXJ9IGZyb20gJy4uL2lvL2luZGV4ZWRfZGInO1xuaW1wb3J0IHtCcm93c2VyTG9jYWxTdG9yYWdlLCBCcm93c2VyTG9jYWxTdG9yYWdlTWFuYWdlcn0gZnJvbSAnLi4vaW8vbG9jYWxfc3RvcmFnZSc7XG5pbXBvcnQge01vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnl9IGZyb20gJy4uL2lvL21vZGVsX21hbmFnZW1lbnQnO1xuXG5pbXBvcnQge1BsYXRmb3JtfSBmcm9tICcuL3BsYXRmb3JtJztcblxuZXhwb3J0IGNsYXNzIFBsYXRmb3JtQnJvd3NlciBpbXBsZW1lbnRzIFBsYXRmb3JtIHtcbiAgLy8gQWNjb3JkaW5nIHRvIHRoZSBzcGVjLCB0aGUgYnVpbHQtaW4gZW5jb2RlciBjYW4gZG8gb25seSBVVEYtOCBlbmNvZGluZy5cbiAgLy8gaHR0cHM6Ly9kZXZlbG9wZXIubW96aWxsYS5vcmcvZW4tVVMvZG9jcy9XZWIvQVBJL1RleHRFbmNvZGVyL1RleHRFbmNvZGVyXG4gIHByaXZhdGUgdGV4dEVuY29kZXI6IFRleHRFbmNvZGVyO1xuXG4gIC8vIEZvciBzZXRUaW1lb3V0Q3VzdG9tXG4gIHByaXZhdGUgcmVhZG9ubHkgbWVzc2FnZU5hbWUgPSAnc2V0VGltZW91dEN1c3RvbSc7XG4gIHByaXZhdGUgZnVuY3Rpb25SZWZzOiBGdW5jdGlvbltdID0gW107XG4gIHByaXZhdGUgaGFuZGxlZE1lc3NhZ2VDb3VudCA9IDA7XG4gIHByaXZhdGUgaGFzRXZlbnRMaXN0ZW5lciA9IGZhbHNlO1xuXG4gIGZldGNoKHBhdGg6IHN0cmluZywgaW5pdD86IFJlcXVlc3RJbml0KTogUHJvbWlzZTxSZXNwb25zZT4ge1xuICAgIHJldHVybiBmZXRjaChwYXRoLCBpbml0KTtcbiAgfVxuXG4gIG5vdygpOiBudW1iZXIge1xuICAgIHJldHVybiBwZXJmb3JtYW5jZS5ub3coKTtcbiAgfVxuXG4gIGVuY29kZSh0ZXh0OiBzdHJpbmcsIGVuY29kaW5nOiBzdHJpbmcpOiBVaW50OEFycmF5IHtcbiAgICBpZiAoZW5jb2RpbmcgIT09ICd1dGYtOCcgJiYgZW5jb2RpbmcgIT09ICd1dGY4Jykge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBCcm93c2VyJ3MgZW5jb2RlciBvbmx5IHN1cHBvcnRzIHV0Zi04LCBidXQgZ290ICR7ZW5jb2Rpbmd9YCk7XG4gICAgfVxuICAgIGlmICh0aGlzLnRleHRFbmNvZGVyID09IG51bGwpIHtcbiAgICAgIHRoaXMudGV4dEVuY29kZXIgPSBuZXcgVGV4dEVuY29kZXIoKTtcbiAgICB9XG4gICAgcmV0dXJuIHRoaXMudGV4dEVuY29kZXIuZW5jb2RlKHRleHQpO1xuICB9XG4gIGRlY29kZShieXRlczogVWludDhBcnJheSwgZW5jb2Rpbmc6IHN0cmluZyk6IHN0cmluZyB7XG4gICAgcmV0dXJuIG5ldyBUZXh0RGVjb2RlcihlbmNvZGluZykuZGVjb2RlKGJ5dGVzKTtcbiAgfVxuXG4gIC8vIElmIHRoZSBzZXRUaW1lb3V0IG5lc3RpbmcgbGV2ZWwgaXMgZ3JlYXRlciB0aGFuIDUgYW5kIHRpbWVvdXQgaXMgbGVzc1xuICAvLyB0aGFuIDRtcywgdGltZW91dCB3aWxsIGJlIGNsYW1wZWQgdG8gNG1zLCB3aGljaCBodXJ0cyB0aGUgcGVyZi5cbiAgLy8gSW50ZXJsZWF2aW5nIHdpbmRvdy5wb3N0TWVzc2FnZSBhbmQgc2V0VGltZW91dCB3aWxsIHRyaWNrIHRoZSBicm93c2VyIGFuZFxuICAvLyBhdm9pZCB0aGUgY2xhbXAuXG4gIHNldFRpbWVvdXRDdXN0b20oZnVuY3Rpb25SZWY6IEZ1bmN0aW9uLCBkZWxheTogbnVtYmVyKTogdm9pZCB7XG4gICAgaWYgKHR5cGVvZiB3aW5kb3cgPT09ICd1bmRlZmluZWQnIHx8XG4gICAgICAgICFlbnYoKS5nZXRCb29sKCdVU0VfU0VUVElNRU9VVENVU1RPTScpKSB7XG4gICAgICBzZXRUaW1lb3V0KGZ1bmN0aW9uUmVmLCBkZWxheSk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgdGhpcy5mdW5jdGlvblJlZnMucHVzaChmdW5jdGlvblJlZik7XG4gICAgc2V0VGltZW91dCgoKSA9PiB7XG4gICAgICB3aW5kb3cucG9zdE1lc3NhZ2UoXG4gICAgICAgICAge25hbWU6IHRoaXMubWVzc2FnZU5hbWUsIGluZGV4OiB0aGlzLmZ1bmN0aW9uUmVmcy5sZW5ndGggLSAxfSwgJyonKTtcbiAgICB9LCBkZWxheSk7XG5cbiAgICBpZiAoIXRoaXMuaGFzRXZlbnRMaXN0ZW5lcikge1xuICAgICAgdGhpcy5oYXNFdmVudExpc3RlbmVyID0gdHJ1ZTtcbiAgICAgIHdpbmRvdy5hZGRFdmVudExpc3RlbmVyKCdtZXNzYWdlJywgKGV2ZW50OiBNZXNzYWdlRXZlbnQpID0+IHtcbiAgICAgICAgaWYgKGV2ZW50LnNvdXJjZSA9PT0gd2luZG93ICYmIGV2ZW50LmRhdGEubmFtZSA9PT0gdGhpcy5tZXNzYWdlTmFtZSkge1xuICAgICAgICAgIGV2ZW50LnN0b3BQcm9wYWdhdGlvbigpO1xuICAgICAgICAgIGNvbnN0IGZ1bmN0aW9uUmVmID0gdGhpcy5mdW5jdGlvblJlZnNbZXZlbnQuZGF0YS5pbmRleF07XG4gICAgICAgICAgZnVuY3Rpb25SZWYoKTtcbiAgICAgICAgICB0aGlzLmhhbmRsZWRNZXNzYWdlQ291bnQrKztcbiAgICAgICAgICBpZiAodGhpcy5oYW5kbGVkTWVzc2FnZUNvdW50ID09PSB0aGlzLmZ1bmN0aW9uUmVmcy5sZW5ndGgpIHtcbiAgICAgICAgICAgIHRoaXMuZnVuY3Rpb25SZWZzID0gW107XG4gICAgICAgICAgICB0aGlzLmhhbmRsZWRNZXNzYWdlQ291bnQgPSAwO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfSwgdHJ1ZSk7XG4gICAgfVxuICB9XG5cbiAgaXNUeXBlZEFycmF5KGE6IHVua25vd24pOiBhIGlzIFVpbnQ4QXJyYXkgfCBGbG9hdDMyQXJyYXkgfCBJbnQzMkFycmF5XG4gICAgfCBVaW50OENsYW1wZWRBcnJheSB7XG4gICAgcmV0dXJuIGEgaW5zdGFuY2VvZiBGbG9hdDMyQXJyYXkgfHwgYSBpbnN0YW5jZW9mIEludDMyQXJyYXkgfHxcbiAgICAgIGEgaW5zdGFuY2VvZiBVaW50OEFycmF5IHx8IGEgaW5zdGFuY2VvZiBVaW50OENsYW1wZWRBcnJheTtcbiAgfVxufVxuXG5pZiAoZW52KCkuZ2V0KCdJU19CUk9XU0VSJykpIHtcbiAgZW52KCkuc2V0UGxhdGZvcm0oJ2Jyb3dzZXInLCBuZXcgUGxhdGZvcm1Ccm93c2VyKCkpO1xuXG4gIC8vIFJlZ2lzdGVyIExvY2FsU3RvcmFnZSBJT0hhbmRsZXJcbiAgdHJ5IHtcbiAgICBNb2RlbFN0b3JlTWFuYWdlclJlZ2lzdHJ5LnJlZ2lzdGVyTWFuYWdlcihcbiAgICAgICAgQnJvd3NlckxvY2FsU3RvcmFnZS5VUkxfU0NIRU1FLCBuZXcgQnJvd3NlckxvY2FsU3RvcmFnZU1hbmFnZXIoKSk7XG4gIH0gY2F0Y2ggKGVycikge1xuICB9XG5cbiAgLy8gUmVnaXN0ZXIgSW5kZXhlZERCIElPSGFuZGxlclxuICB0cnkge1xuICAgIE1vZGVsU3RvcmVNYW5hZ2VyUmVnaXN0cnkucmVnaXN0ZXJNYW5hZ2VyKFxuICAgICAgICBCcm93c2VySW5kZXhlZERCLlVSTF9TQ0hFTUUsIG5ldyBCcm93c2VySW5kZXhlZERCTWFuYWdlcigpKTtcbiAgfSBjYXRjaCAoZXJyKSB7XG4gIH1cbn1cbiJdfQ==