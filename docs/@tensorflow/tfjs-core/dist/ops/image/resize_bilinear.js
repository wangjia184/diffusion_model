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
import { ResizeBilinear } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * Bilinear resize a single 3D image or a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to `false`. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assume pixel centers
 *     are at 0.5, which would make the floating point coordinates of the top
 *     left pixel 0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeBilinear_(images, size, alignCorners = false, halfPixelCenters = false) {
    const $images = convertToTensor(images, 'images', 'resizeBilinear');
    util.assert($images.rank === 3 || $images.rank === 4, () => `Error in resizeBilinear: x must be rank 3 or 4, but got ` +
        `rank ${$images.rank}.`);
    util.assert(size.length === 2, () => `Error in resizeBilinear: new shape must 2D, but got shape ` +
        `${size}.`);
    util.assert(halfPixelCenters === false || alignCorners === false, () => `Error in resizeBilinear: If halfPixelCenters is true, ` +
        `alignCorners must be false.`);
    let batchImages = $images;
    let reshapedTo4D = false;
    if ($images.rank === 3) {
        reshapedTo4D = true;
        batchImages = reshape($images, [1, $images.shape[0], $images.shape[1], $images.shape[2]]);
    }
    const [] = size;
    const inputs = { images: batchImages };
    const attrs = { alignCorners, halfPixelCenters, size };
    // tslint:disable-next-line: no-unnecessary-type-assertion
    const res = ENGINE.runKernel(ResizeBilinear, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const resizeBilinear = /* @__PURE__ */ op({ resizeBilinear_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX2JpbGluZWFyLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2UvcmVzaXplX2JpbGluZWFyLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGNBQWMsRUFBNEMsTUFBTSxvQkFBb0IsQ0FBQztBQUk3RixPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DOzs7Ozs7Ozs7Ozs7Ozs7O0dBZ0JHO0FBQ0gsU0FBUyxlQUFlLENBQ3BCLE1BQW9CLEVBQUUsSUFBc0IsRUFBRSxZQUFZLEdBQUcsS0FBSyxFQUNsRSxnQkFBZ0IsR0FBRyxLQUFLO0lBQzFCLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLGdCQUFnQixDQUFDLENBQUM7SUFFcEUsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDeEMsR0FBRyxFQUFFLENBQUMsMERBQTBEO1FBQzVELFFBQVEsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDakIsR0FBRyxFQUFFLENBQUMsNERBQTREO1FBQzlELEdBQUcsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNwQixJQUFJLENBQUMsTUFBTSxDQUNQLGdCQUFnQixLQUFLLEtBQUssSUFBSSxZQUFZLEtBQUssS0FBSyxFQUNwRCxHQUFHLEVBQUUsQ0FBQyx3REFBd0Q7UUFDMUQsNkJBQTZCLENBQUMsQ0FBQztJQUV2QyxJQUFJLFdBQVcsR0FBRyxPQUFtQixDQUFDO0lBQ3RDLElBQUksWUFBWSxHQUFHLEtBQUssQ0FBQztJQUN6QixJQUFJLE9BQU8sQ0FBQyxJQUFJLEtBQUssQ0FBQyxFQUFFO1FBQ3RCLFlBQVksR0FBRyxJQUFJLENBQUM7UUFDcEIsV0FBVyxHQUFHLE9BQU8sQ0FDakIsT0FBTyxFQUFFLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQztLQUN6RTtJQUVELE1BQU0sRUFBRSxHQUFHLElBQUksQ0FBQztJQUVoQixNQUFNLE1BQU0sR0FBeUIsRUFBQyxNQUFNLEVBQUUsV0FBVyxFQUFDLENBQUM7SUFDM0QsTUFBTSxLQUFLLEdBQXdCLEVBQUMsWUFBWSxFQUFFLGdCQUFnQixFQUFFLElBQUksRUFBQyxDQUFDO0lBRTFFLDBEQUEwRDtJQUMxRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUMsU0FBUyxDQUNaLGNBQWMsRUFBRSxNQUFtQyxFQUNuRCxLQUFnQyxDQUFNLENBQUM7SUFFdkQsSUFBSSxZQUFZLEVBQUU7UUFDaEIsT0FBTyxPQUFPLENBQUMsR0FBRyxFQUFFLENBQUMsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBTSxDQUFDO0tBQ3RFO0lBQ0QsT0FBTyxHQUFHLENBQUM7QUFDYixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sY0FBYyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxlQUFlLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge0VOR0lORX0gZnJvbSAnLi4vLi4vZW5naW5lJztcbmltcG9ydCB7UmVzaXplQmlsaW5lYXIsIFJlc2l6ZUJpbGluZWFyQXR0cnMsIFJlc2l6ZUJpbGluZWFySW5wdXRzfSBmcm9tICcuLi8uLi9rZXJuZWxfbmFtZXMnO1xuaW1wb3J0IHtOYW1lZEF0dHJNYXB9IGZyb20gJy4uLy4uL2tlcm5lbF9yZWdpc3RyeSc7XG5pbXBvcnQge1RlbnNvcjNELCBUZW5zb3I0RH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7TmFtZWRUZW5zb3JNYXB9IGZyb20gJy4uLy4uL3RlbnNvcl90eXBlcyc7XG5pbXBvcnQge2NvbnZlcnRUb1RlbnNvcn0gZnJvbSAnLi4vLi4vdGVuc29yX3V0aWxfZW52JztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vLi4vdHlwZXMnO1xuaW1wb3J0ICogYXMgdXRpbCBmcm9tICcuLi8uLi91dGlsJztcblxuaW1wb3J0IHtvcH0gZnJvbSAnLi4vb3BlcmF0aW9uJztcbmltcG9ydCB7cmVzaGFwZX0gZnJvbSAnLi4vcmVzaGFwZSc7XG5cbi8qKlxuICogQmlsaW5lYXIgcmVzaXplIGEgc2luZ2xlIDNEIGltYWdlIG9yIGEgYmF0Y2ggb2YgM0QgaW1hZ2VzIHRvIGEgbmV3IHNoYXBlLlxuICpcbiAqIEBwYXJhbSBpbWFnZXMgVGhlIGltYWdlcywgb2YgcmFuayA0IG9yIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpcyBhc3N1bWVkLlxuICogQHBhcmFtIHNpemUgVGhlIG5ldyBzaGFwZSBgW25ld0hlaWdodCwgbmV3V2lkdGhdYCB0byByZXNpemUgdGhlXG4gKiAgICAgaW1hZ2VzIHRvLiBFYWNoIGNoYW5uZWwgaXMgcmVzaXplZCBpbmRpdmlkdWFsbHkuXG4gKiBAcGFyYW0gYWxpZ25Db3JuZXJzIERlZmF1bHRzIHRvIGBmYWxzZWAuIElmIHRydWUsIHJlc2NhbGVcbiAqICAgICBpbnB1dCBieSBgKG5ld19oZWlnaHQgLSAxKSAvIChoZWlnaHQgLSAxKWAsIHdoaWNoIGV4YWN0bHkgYWxpZ25zIHRoZSA0XG4gKiAgICAgY29ybmVycyBvZiBpbWFnZXMgYW5kIHJlc2l6ZWQgaW1hZ2VzLiBJZiBmYWxzZSwgcmVzY2FsZSBieVxuICogICAgIGBuZXdfaGVpZ2h0IC8gaGVpZ2h0YC4gVHJlYXQgc2ltaWxhcmx5IHRoZSB3aWR0aCBkaW1lbnNpb24uXG4gKiBAcGFyYW0gaGFsZlBpeGVsQ2VudGVycyBEZWZhdWx0cyB0byBgZmFsc2VgLiBXaGV0aGVyIHRvIGFzc3VtZSBwaXhlbCBjZW50ZXJzXG4gKiAgICAgYXJlIGF0IDAuNSwgd2hpY2ggd291bGQgbWFrZSB0aGUgZmxvYXRpbmcgcG9pbnQgY29vcmRpbmF0ZXMgb2YgdGhlIHRvcFxuICogICAgIGxlZnQgcGl4ZWwgMC41LCAwLjUuXG4gKlxuICogQGRvYyB7aGVhZGluZzogJ09wZXJhdGlvbnMnLCBzdWJoZWFkaW5nOiAnSW1hZ2VzJywgbmFtZXNwYWNlOiAnaW1hZ2UnfVxuICovXG5mdW5jdGlvbiByZXNpemVCaWxpbmVhcl88VCBleHRlbmRzIFRlbnNvcjNEfFRlbnNvcjREPihcbiAgICBpbWFnZXM6IFR8VGVuc29yTGlrZSwgc2l6ZTogW251bWJlciwgbnVtYmVyXSwgYWxpZ25Db3JuZXJzID0gZmFsc2UsXG4gICAgaGFsZlBpeGVsQ2VudGVycyA9IGZhbHNlKTogVCB7XG4gIGNvbnN0ICRpbWFnZXMgPSBjb252ZXJ0VG9UZW5zb3IoaW1hZ2VzLCAnaW1hZ2VzJywgJ3Jlc2l6ZUJpbGluZWFyJyk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkaW1hZ2VzLnJhbmsgPT09IDMgfHwgJGltYWdlcy5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyOiB4IG11c3QgYmUgcmFuayAzIG9yIDQsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYHJhbmsgJHskaW1hZ2VzLnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIHNpemUubGVuZ3RoID09PSAyLFxuICAgICAgKCkgPT4gYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyOiBuZXcgc2hhcGUgbXVzdCAyRCwgYnV0IGdvdCBzaGFwZSBgICtcbiAgICAgICAgICBgJHtzaXplfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBoYWxmUGl4ZWxDZW50ZXJzID09PSBmYWxzZSB8fCBhbGlnbkNvcm5lcnMgPT09IGZhbHNlLFxuICAgICAgKCkgPT4gYEVycm9yIGluIHJlc2l6ZUJpbGluZWFyOiBJZiBoYWxmUGl4ZWxDZW50ZXJzIGlzIHRydWUsIGAgK1xuICAgICAgICAgIGBhbGlnbkNvcm5lcnMgbXVzdCBiZSBmYWxzZS5gKTtcblxuICBsZXQgYmF0Y2hJbWFnZXMgPSAkaW1hZ2VzIGFzIFRlbnNvcjREO1xuICBsZXQgcmVzaGFwZWRUbzREID0gZmFsc2U7XG4gIGlmICgkaW1hZ2VzLnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIGJhdGNoSW1hZ2VzID0gcmVzaGFwZShcbiAgICAgICAgJGltYWdlcywgWzEsICRpbWFnZXMuc2hhcGVbMF0sICRpbWFnZXMuc2hhcGVbMV0sICRpbWFnZXMuc2hhcGVbMl1dKTtcbiAgfVxuXG4gIGNvbnN0IFtdID0gc2l6ZTtcblxuICBjb25zdCBpbnB1dHM6IFJlc2l6ZUJpbGluZWFySW5wdXRzID0ge2ltYWdlczogYmF0Y2hJbWFnZXN9O1xuICBjb25zdCBhdHRyczogUmVzaXplQmlsaW5lYXJBdHRycyA9IHthbGlnbkNvcm5lcnMsIGhhbGZQaXhlbENlbnRlcnMsIHNpemV9O1xuXG4gIC8vIHRzbGludDpkaXNhYmxlLW5leHQtbGluZTogbm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgY29uc3QgcmVzID0gRU5HSU5FLnJ1bktlcm5lbChcbiAgICAgICAgICAgICAgICAgIFJlc2l6ZUJpbGluZWFyLCBpbnB1dHMgYXMgdW5rbm93biBhcyBOYW1lZFRlbnNvck1hcCxcbiAgICAgICAgICAgICAgICAgIGF0dHJzIGFzIHVua25vd24gYXMgTmFtZWRBdHRyTWFwKSBhcyBUO1xuXG4gIGlmIChyZXNoYXBlZFRvNEQpIHtcbiAgICByZXR1cm4gcmVzaGFwZShyZXMsIFtyZXMuc2hhcGVbMV0sIHJlcy5zaGFwZVsyXSwgcmVzLnNoYXBlWzNdXSkgYXMgVDtcbiAgfVxuICByZXR1cm4gcmVzO1xufVxuXG5leHBvcnQgY29uc3QgcmVzaXplQmlsaW5lYXIgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Jlc2l6ZUJpbGluZWFyX30pO1xuIl19