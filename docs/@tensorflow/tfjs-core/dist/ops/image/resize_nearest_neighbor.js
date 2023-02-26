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
import { ResizeNearestNeighbor } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
import { reshape } from '../reshape';
/**
 * NearestNeighbor resize a batch of 3D images to a new shape.
 *
 * @param images The images, of rank 4 or rank 3, of shape
 *     `[batch, height, width, inChannels]`. If rank 3, batch of 1 is assumed.
 * @param size The new shape `[newHeight, newWidth]` to resize the
 *     images to. Each channel is resized individually.
 * @param alignCorners Defaults to False. If true, rescale
 *     input by `(new_height - 1) / (height - 1)`, which exactly aligns the 4
 *     corners of images and resized images. If false, rescale by
 *     `new_height / height`. Treat similarly the width dimension.
 * @param halfPixelCenters Defaults to `false`. Whether to assume pixels are of
 *      half the actual dimensions, and yield more accurate resizes. This flag
 *      would also make the floating point coordinates of the top left pixel
 *      0.5, 0.5.
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function resizeNearestNeighbor_(images, size, alignCorners = false, halfPixelCenters = false) {
    const $images = convertToTensor(images, 'images', 'resizeNearestNeighbor');
    util.assert($images.rank === 3 || $images.rank === 4, () => `Error in resizeNearestNeighbor: x must be rank 3 or 4, but got ` +
        `rank ${$images.rank}.`);
    util.assert(size.length === 2, () => `Error in resizeNearestNeighbor: new shape must 2D, but got shape ` +
        `${size}.`);
    util.assert($images.dtype === 'float32' || $images.dtype === 'int32', () => '`images` must have `int32` or `float32` as dtype');
    util.assert(halfPixelCenters === false || alignCorners === false, () => `Error in resizeNearestNeighbor: If halfPixelCenters is true, ` +
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
    const res = ENGINE.runKernel(ResizeNearestNeighbor, inputs, attrs);
    if (reshapedTo4D) {
        return reshape(res, [res.shape[1], res.shape[2], res.shape[3]]);
    }
    return res;
}
export const resizeNearestNeighbor = /* @__PURE__ */ op({ resizeNearestNeighbor_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVzaXplX25lYXJlc3RfbmVpZ2hib3IuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9pbWFnZS9yZXNpemVfbmVhcmVzdF9uZWlnaGJvci50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7Ozs7Ozs7O0dBZUc7QUFFSCxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sY0FBYyxDQUFDO0FBQ3BDLE9BQU8sRUFBQyxxQkFBcUIsRUFBMEQsTUFBTSxvQkFBb0IsQ0FBQztBQUlsSCxPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUNoQyxPQUFPLEVBQUMsT0FBTyxFQUFDLE1BQU0sWUFBWSxDQUFDO0FBRW5DOzs7Ozs7Ozs7Ozs7Ozs7OztHQWlCRztBQUNILFNBQVMsc0JBQXNCLENBQzNCLE1BQW9CLEVBQUUsSUFBc0IsRUFBRSxZQUFZLEdBQUcsS0FBSyxFQUNsRSxnQkFBZ0IsR0FBRyxLQUFLO0lBQzFCLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLHVCQUF1QixDQUFDLENBQUM7SUFFM0UsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsRUFDeEMsR0FBRyxFQUFFLENBQUMsaUVBQWlFO1FBQ25FLFFBQVEsT0FBTyxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDakMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxJQUFJLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDakIsR0FBRyxFQUFFLENBQ0QsbUVBQW1FO1FBQ25FLEdBQUcsSUFBSSxHQUFHLENBQUMsQ0FBQztJQUNwQixJQUFJLENBQUMsTUFBTSxDQUNQLE9BQU8sQ0FBQyxLQUFLLEtBQUssU0FBUyxJQUFJLE9BQU8sQ0FBQyxLQUFLLEtBQUssT0FBTyxFQUN4RCxHQUFHLEVBQUUsQ0FBQyxrREFBa0QsQ0FBQyxDQUFDO0lBQzlELElBQUksQ0FBQyxNQUFNLENBQ1AsZ0JBQWdCLEtBQUssS0FBSyxJQUFJLFlBQVksS0FBSyxLQUFLLEVBQ3BELEdBQUcsRUFBRSxDQUFDLCtEQUErRDtRQUNqRSw2QkFBNkIsQ0FBQyxDQUFDO0lBQ3ZDLElBQUksV0FBVyxHQUFHLE9BQW1CLENBQUM7SUFDdEMsSUFBSSxZQUFZLEdBQUcsS0FBSyxDQUFDO0lBQ3pCLElBQUksT0FBTyxDQUFDLElBQUksS0FBSyxDQUFDLEVBQUU7UUFDdEIsWUFBWSxHQUFHLElBQUksQ0FBQztRQUNwQixXQUFXLEdBQUcsT0FBTyxDQUNqQixPQUFPLEVBQUUsQ0FBQyxDQUFDLEVBQUUsT0FBTyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxFQUFFLE9BQU8sQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO0tBQ3pFO0lBQ0QsTUFBTSxFQUFFLEdBQUcsSUFBSSxDQUFDO0lBRWhCLE1BQU0sTUFBTSxHQUFnQyxFQUFDLE1BQU0sRUFBRSxXQUFXLEVBQUMsQ0FBQztJQUNsRSxNQUFNLEtBQUssR0FDc0IsRUFBQyxZQUFZLEVBQUUsZ0JBQWdCLEVBQUUsSUFBSSxFQUFDLENBQUM7SUFFeEUsMERBQTBEO0lBQzFELE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ1oscUJBQXFCLEVBQUUsTUFBbUMsRUFDMUQsS0FBZ0MsQ0FBTSxDQUFDO0lBRXZELElBQUksWUFBWSxFQUFFO1FBQ2hCLE9BQU8sT0FBTyxDQUFDLEdBQUcsRUFBRSxDQUFDLEdBQUcsQ0FBQyxLQUFLLENBQUMsQ0FBQyxDQUFDLEVBQUUsR0FBRyxDQUFDLEtBQUssQ0FBQyxDQUFDLENBQUMsRUFBRSxHQUFHLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQU0sQ0FBQztLQUN0RTtJQUNELE9BQU8sR0FBRyxDQUFDO0FBQ2IsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLHFCQUFxQixHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxzQkFBc0IsRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAyMCBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7RU5HSU5FfSBmcm9tICcuLi8uLi9lbmdpbmUnO1xuaW1wb3J0IHtSZXNpemVOZWFyZXN0TmVpZ2hib3IsIFJlc2l6ZU5lYXJlc3ROZWlnaGJvckF0dHJzLCBSZXNpemVOZWFyZXN0TmVpZ2hib3JJbnB1dHN9IGZyb20gJy4uLy4uL2tlcm5lbF9uYW1lcyc7XG5pbXBvcnQge05hbWVkQXR0ck1hcH0gZnJvbSAnLi4vLi4va2VybmVsX3JlZ2lzdHJ5JztcbmltcG9ydCB7VGVuc29yM0QsIFRlbnNvcjREfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuaW1wb3J0IHtyZXNoYXBlfSBmcm9tICcuLi9yZXNoYXBlJztcblxuLyoqXG4gKiBOZWFyZXN0TmVpZ2hib3IgcmVzaXplIGEgYmF0Y2ggb2YgM0QgaW1hZ2VzIHRvIGEgbmV3IHNoYXBlLlxuICpcbiAqIEBwYXJhbSBpbWFnZXMgVGhlIGltYWdlcywgb2YgcmFuayA0IG9yIHJhbmsgMywgb2Ygc2hhcGVcbiAqICAgICBgW2JhdGNoLCBoZWlnaHQsIHdpZHRoLCBpbkNoYW5uZWxzXWAuIElmIHJhbmsgMywgYmF0Y2ggb2YgMSBpcyBhc3N1bWVkLlxuICogQHBhcmFtIHNpemUgVGhlIG5ldyBzaGFwZSBgW25ld0hlaWdodCwgbmV3V2lkdGhdYCB0byByZXNpemUgdGhlXG4gKiAgICAgaW1hZ2VzIHRvLiBFYWNoIGNoYW5uZWwgaXMgcmVzaXplZCBpbmRpdmlkdWFsbHkuXG4gKiBAcGFyYW0gYWxpZ25Db3JuZXJzIERlZmF1bHRzIHRvIEZhbHNlLiBJZiB0cnVlLCByZXNjYWxlXG4gKiAgICAgaW5wdXQgYnkgYChuZXdfaGVpZ2h0IC0gMSkgLyAoaGVpZ2h0IC0gMSlgLCB3aGljaCBleGFjdGx5IGFsaWducyB0aGUgNFxuICogICAgIGNvcm5lcnMgb2YgaW1hZ2VzIGFuZCByZXNpemVkIGltYWdlcy4gSWYgZmFsc2UsIHJlc2NhbGUgYnlcbiAqICAgICBgbmV3X2hlaWdodCAvIGhlaWdodGAuIFRyZWF0IHNpbWlsYXJseSB0aGUgd2lkdGggZGltZW5zaW9uLlxuICogQHBhcmFtIGhhbGZQaXhlbENlbnRlcnMgRGVmYXVsdHMgdG8gYGZhbHNlYC4gV2hldGhlciB0byBhc3N1bWUgcGl4ZWxzIGFyZSBvZlxuICogICAgICBoYWxmIHRoZSBhY3R1YWwgZGltZW5zaW9ucywgYW5kIHlpZWxkIG1vcmUgYWNjdXJhdGUgcmVzaXplcy4gVGhpcyBmbGFnXG4gKiAgICAgIHdvdWxkIGFsc28gbWFrZSB0aGUgZmxvYXRpbmcgcG9pbnQgY29vcmRpbmF0ZXMgb2YgdGhlIHRvcCBsZWZ0IHBpeGVsXG4gKiAgICAgIDAuNSwgMC41LlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0ltYWdlcycsIG5hbWVzcGFjZTogJ2ltYWdlJ31cbiAqL1xuZnVuY3Rpb24gcmVzaXplTmVhcmVzdE5laWdoYm9yXzxUIGV4dGVuZHMgVGVuc29yM0R8VGVuc29yNEQ+KFxuICAgIGltYWdlczogVHxUZW5zb3JMaWtlLCBzaXplOiBbbnVtYmVyLCBudW1iZXJdLCBhbGlnbkNvcm5lcnMgPSBmYWxzZSxcbiAgICBoYWxmUGl4ZWxDZW50ZXJzID0gZmFsc2UpOiBUIHtcbiAgY29uc3QgJGltYWdlcyA9IGNvbnZlcnRUb1RlbnNvcihpbWFnZXMsICdpbWFnZXMnLCAncmVzaXplTmVhcmVzdE5laWdoYm9yJyk7XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkaW1hZ2VzLnJhbmsgPT09IDMgfHwgJGltYWdlcy5yYW5rID09PSA0LFxuICAgICAgKCkgPT4gYEVycm9yIGluIHJlc2l6ZU5lYXJlc3ROZWlnaGJvcjogeCBtdXN0IGJlIHJhbmsgMyBvciA0LCBidXQgZ290IGAgK1xuICAgICAgICAgIGByYW5rICR7JGltYWdlcy5yYW5rfS5gKTtcbiAgdXRpbC5hc3NlcnQoXG4gICAgICBzaXplLmxlbmd0aCA9PT0gMixcbiAgICAgICgpID0+XG4gICAgICAgICAgYEVycm9yIGluIHJlc2l6ZU5lYXJlc3ROZWlnaGJvcjogbmV3IHNoYXBlIG11c3QgMkQsIGJ1dCBnb3Qgc2hhcGUgYCArXG4gICAgICAgICAgYCR7c2l6ZX0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgJGltYWdlcy5kdHlwZSA9PT0gJ2Zsb2F0MzInIHx8ICRpbWFnZXMuZHR5cGUgPT09ICdpbnQzMicsXG4gICAgICAoKSA9PiAnYGltYWdlc2AgbXVzdCBoYXZlIGBpbnQzMmAgb3IgYGZsb2F0MzJgIGFzIGR0eXBlJyk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgaGFsZlBpeGVsQ2VudGVycyA9PT0gZmFsc2UgfHwgYWxpZ25Db3JuZXJzID09PSBmYWxzZSxcbiAgICAgICgpID0+IGBFcnJvciBpbiByZXNpemVOZWFyZXN0TmVpZ2hib3I6IElmIGhhbGZQaXhlbENlbnRlcnMgaXMgdHJ1ZSwgYCArXG4gICAgICAgICAgYGFsaWduQ29ybmVycyBtdXN0IGJlIGZhbHNlLmApO1xuICBsZXQgYmF0Y2hJbWFnZXMgPSAkaW1hZ2VzIGFzIFRlbnNvcjREO1xuICBsZXQgcmVzaGFwZWRUbzREID0gZmFsc2U7XG4gIGlmICgkaW1hZ2VzLnJhbmsgPT09IDMpIHtcbiAgICByZXNoYXBlZFRvNEQgPSB0cnVlO1xuICAgIGJhdGNoSW1hZ2VzID0gcmVzaGFwZShcbiAgICAgICAgJGltYWdlcywgWzEsICRpbWFnZXMuc2hhcGVbMF0sICRpbWFnZXMuc2hhcGVbMV0sICRpbWFnZXMuc2hhcGVbMl1dKTtcbiAgfVxuICBjb25zdCBbXSA9IHNpemU7XG5cbiAgY29uc3QgaW5wdXRzOiBSZXNpemVOZWFyZXN0TmVpZ2hib3JJbnB1dHMgPSB7aW1hZ2VzOiBiYXRjaEltYWdlc307XG4gIGNvbnN0IGF0dHJzOlxuICAgICAgUmVzaXplTmVhcmVzdE5laWdoYm9yQXR0cnMgPSB7YWxpZ25Db3JuZXJzLCBoYWxmUGl4ZWxDZW50ZXJzLCBzaXplfTtcblxuICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6IG5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gIGNvbnN0IHJlcyA9IEVOR0lORS5ydW5LZXJuZWwoXG4gICAgICAgICAgICAgICAgICBSZXNpemVOZWFyZXN0TmVpZ2hib3IsIGlucHV0cyBhcyB1bmtub3duIGFzIE5hbWVkVGVuc29yTWFwLFxuICAgICAgICAgICAgICAgICAgYXR0cnMgYXMgdW5rbm93biBhcyBOYW1lZEF0dHJNYXApIGFzIFQ7XG5cbiAgaWYgKHJlc2hhcGVkVG80RCkge1xuICAgIHJldHVybiByZXNoYXBlKHJlcywgW3Jlcy5zaGFwZVsxXSwgcmVzLnNoYXBlWzJdLCByZXMuc2hhcGVbM11dKSBhcyBUO1xuICB9XG4gIHJldHVybiByZXM7XG59XG5cbmV4cG9ydCBjb25zdCByZXNpemVOZWFyZXN0TmVpZ2hib3IgPSAvKiBAX19QVVJFX18gKi8gb3Aoe3Jlc2l6ZU5lYXJlc3ROZWlnaGJvcl99KTtcbiJdfQ==