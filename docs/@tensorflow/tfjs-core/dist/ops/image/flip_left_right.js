/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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
import { ENGINE } from '../../engine';
import { FlipLeftRight } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
/**
 * Flips the image left to right. Currently available in the CPU, WebGL, and
 * WASM backends.
 *
 * @param image 4d tensor of shape `[batch, imageHeight, imageWidth, depth]`.
 */
/** @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'} */
function flipLeftRight_(image) {
    const $image = convertToTensor(image, 'image', 'flipLeftRight', 'float32');
    util.assert($image.rank === 4, () => 'Error in flipLeftRight: image must be rank 4,' +
        `but got rank ${$image.rank}.`);
    const inputs = { image: $image };
    const res = ENGINE.runKernel(FlipLeftRight, inputs, {});
    return res;
}
export const flipLeftRight = /* @__PURE__ */ op({ flipLeftRight_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZmxpcF9sZWZ0X3JpZ2h0LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2UvZmxpcF9sZWZ0X3JpZ2h0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGFBQWEsRUFBc0IsTUFBTSxvQkFBb0IsQ0FBQztBQUd0RSxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFDbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7R0FLRztBQUNILDZFQUE2RTtBQUM3RSxTQUFTLGNBQWMsQ0FBQyxLQUEwQjtJQUNoRCxNQUFNLE1BQU0sR0FBRyxlQUFlLENBQUMsS0FBSyxFQUFFLE9BQU8sRUFBRSxlQUFlLEVBQUUsU0FBUyxDQUFDLENBQUM7SUFFM0UsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDakIsR0FBRyxFQUFFLENBQUMsK0NBQStDO1FBQ2pELGdCQUFnQixNQUFNLENBQUMsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUV4QyxNQUFNLE1BQU0sR0FBd0IsRUFBQyxLQUFLLEVBQUUsTUFBTSxFQUFDLENBQUM7SUFDcEQsTUFBTSxHQUFHLEdBQ0wsTUFBTSxDQUFDLFNBQVMsQ0FBQyxhQUFhLEVBQUUsTUFBbUMsRUFBRSxFQUFFLENBQUMsQ0FBQztJQUM3RSxPQUFPLEdBQWUsQ0FBQztBQUN6QixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sYUFBYSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxjQUFjLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7RmxpcExlZnRSaWdodCwgRmxpcExlZnRSaWdodElucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7VGVuc29yNER9IGZyb20gJy4uLy4uL3RlbnNvcic7XG5pbXBvcnQge05hbWVkVGVuc29yTWFwfSBmcm9tICcuLi8uLi90ZW5zb3JfdHlwZXMnO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uLy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uLy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vLi4vdXRpbCc7XG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEZsaXBzIHRoZSBpbWFnZSBsZWZ0IHRvIHJpZ2h0LiBDdXJyZW50bHkgYXZhaWxhYmxlIGluIHRoZSBDUFUsIFdlYkdMLCBhbmRcbiAqIFdBU00gYmFja2VuZHMuXG4gKlxuICogQHBhcmFtIGltYWdlIDRkIHRlbnNvciBvZiBzaGFwZSBgW2JhdGNoLCBpbWFnZUhlaWdodCwgaW1hZ2VXaWR0aCwgZGVwdGhdYC5cbiAqL1xuLyoqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0ltYWdlcycsIG5hbWVzcGFjZTogJ2ltYWdlJ30gKi9cbmZ1bmN0aW9uIGZsaXBMZWZ0UmlnaHRfKGltYWdlOiBUZW5zb3I0RHxUZW5zb3JMaWtlKTogVGVuc29yNEQge1xuICBjb25zdCAkaW1hZ2UgPSBjb252ZXJ0VG9UZW5zb3IoaW1hZ2UsICdpbWFnZScsICdmbGlwTGVmdFJpZ2h0JywgJ2Zsb2F0MzInKTtcblxuICB1dGlsLmFzc2VydChcbiAgICAgICRpbWFnZS5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gJ0Vycm9yIGluIGZsaXBMZWZ0UmlnaHQ6IGltYWdlIG11c3QgYmUgcmFuayA0LCcgK1xuICAgICAgICAgIGBidXQgZ290IHJhbmsgJHskaW1hZ2UucmFua30uYCk7XG5cbiAgY29uc3QgaW5wdXRzOiBGbGlwTGVmdFJpZ2h0SW5wdXRzID0ge2ltYWdlOiAkaW1hZ2V9O1xuICBjb25zdCByZXMgPVxuICAgICAgRU5HSU5FLnJ1bktlcm5lbChGbGlwTGVmdFJpZ2h0LCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCwge30pO1xuICByZXR1cm4gcmVzIGFzIFRlbnNvcjREO1xufVxuXG5leHBvcnQgY29uc3QgZmxpcExlZnRSaWdodCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZmxpcExlZnRSaWdodF99KTtcbiJdfQ==