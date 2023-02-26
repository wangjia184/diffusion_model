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
import { CropAndResize } from '../../kernel_names';
import { convertToTensor } from '../../tensor_util_env';
import * as util from '../../util';
import { op } from '../operation';
/**
 * Extracts crops from the input image tensor and resizes them using bilinear
 * sampling or nearest neighbor sampling (possibly with aspect ratio change)
 * to a common output size specified by cropSize.
 *
 * @param image 4d tensor of shape `[batch,imageHeight,imageWidth, depth]`,
 *     where imageHeight and imageWidth must be positive, specifying the
 *     batch of images from which to take crops
 * @param boxes 2d float32 tensor of shape `[numBoxes, 4]`. Each entry is
 *     `[y1, x1, y2, x2]`, where `(y1, x1)` and `(y2, x2)` are the normalized
 *     coordinates of the box in the `boxInd[i]`th image in the batch
 * @param boxInd 1d int32 tensor of shape `[numBoxes]` with values in range
 *     `[0, batch)` that specifies the image that the `i`-th box refers to.
 * @param cropSize 1d int32 tensor of 2 elements `[cropHeigh, cropWidth]`
 *     specifying the size to which all crops are resized to.
 * @param method Optional string from `'bilinear' | 'nearest'`,
 *     defaults to bilinear, which specifies the sampling method for resizing
 * @param extrapolationValue A threshold for deciding when to remove boxes based
 *     on score. Defaults to 0.
 * @return A 4D tensor of the shape `[numBoxes,cropHeight,cropWidth,depth]`
 *
 * @doc {heading: 'Operations', subheading: 'Images', namespace: 'image'}
 */
function cropAndResize_(image, boxes, boxInd, cropSize, method = 'bilinear', extrapolationValue = 0) {
    const $image = convertToTensor(image, 'image', 'cropAndResize');
    const $boxes = convertToTensor(boxes, 'boxes', 'cropAndResize', 'float32');
    const $boxInd = convertToTensor(boxInd, 'boxInd', 'cropAndResize', 'int32');
    const numBoxes = $boxes.shape[0];
    util.assert($image.rank === 4, () => 'Error in cropAndResize: image must be rank 4,' +
        `but got rank ${$image.rank}.`);
    util.assert($boxes.rank === 2 && $boxes.shape[1] === 4, () => `Error in cropAndResize: boxes must be have size [${numBoxes},4] ` +
        `but had shape ${$boxes.shape}.`);
    util.assert($boxInd.rank === 1 && $boxInd.shape[0] === numBoxes, () => `Error in cropAndResize: boxInd must be have size [${numBoxes}] ` +
        `but had shape ${$boxes.shape}.`);
    util.assert(cropSize.length === 2, () => `Error in cropAndResize: cropSize must be of length 2, but got ` +
        `length ${cropSize.length}.`);
    util.assert(cropSize[0] >= 1 && cropSize[1] >= 1, () => `cropSize must be atleast [1,1], but was ${cropSize}`);
    util.assert(method === 'bilinear' || method === 'nearest', () => `method must be bilinear or nearest, but was ${method}`);
    const inputs = { image: $image, boxes: $boxes, boxInd: $boxInd };
    const attrs = { method, extrapolationValue, cropSize };
    const res = ENGINE.runKernel(CropAndResize, inputs, attrs);
    return res;
}
export const cropAndResize = /* @__PURE__ */ op({ cropAndResize_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY3JvcF9hbmRfcmVzaXplLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvaW1hZ2UvY3JvcF9hbmRfcmVzaXplLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUVILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDcEMsT0FBTyxFQUFDLGFBQWEsRUFBMEMsTUFBTSxvQkFBb0IsQ0FBQztBQUkxRixPQUFPLEVBQUMsZUFBZSxFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFdEQsT0FBTyxLQUFLLElBQUksTUFBTSxZQUFZLENBQUM7QUFFbkMsT0FBTyxFQUFDLEVBQUUsRUFBQyxNQUFNLGNBQWMsQ0FBQztBQUVoQzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQXNCRztBQUNILFNBQVMsY0FBYyxDQUNuQixLQUEwQixFQUMxQixLQUEwQixFQUMxQixNQUEyQixFQUMzQixRQUEwQixFQUMxQixTQUErQixVQUFVLEVBQ3pDLGtCQUFrQixHQUFHLENBQUM7SUFFeEIsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZUFBZSxDQUFDLENBQUM7SUFDaEUsTUFBTSxNQUFNLEdBQUcsZUFBZSxDQUFDLEtBQUssRUFBRSxPQUFPLEVBQUUsZUFBZSxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQzNFLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxNQUFNLEVBQUUsUUFBUSxFQUFFLGVBQWUsRUFBRSxPQUFPLENBQUMsQ0FBQztJQUU1RSxNQUFNLFFBQVEsR0FBRyxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxDQUFDO0lBRWpDLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2pCLEdBQUcsRUFBRSxDQUFDLCtDQUErQztRQUNqRCxnQkFBZ0IsTUFBTSxDQUFDLElBQUksR0FBRyxDQUFDLENBQUM7SUFDeEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxNQUFNLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxNQUFNLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLENBQUMsRUFDMUMsR0FBRyxFQUFFLENBQUMsb0RBQW9ELFFBQVEsTUFBTTtRQUNwRSxpQkFBaUIsTUFBTSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDMUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxPQUFPLENBQUMsSUFBSSxLQUFLLENBQUMsSUFBSSxPQUFPLENBQUMsS0FBSyxDQUFDLENBQUMsQ0FBQyxLQUFLLFFBQVEsRUFDbkQsR0FBRyxFQUFFLENBQUMscURBQXFELFFBQVEsSUFBSTtRQUNuRSxpQkFBaUIsTUFBTSxDQUFDLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDMUMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsTUFBTSxLQUFLLENBQUMsRUFDckIsR0FBRyxFQUFFLENBQUMsZ0VBQWdFO1FBQ2xFLFVBQVUsUUFBUSxDQUFDLE1BQU0sR0FBRyxDQUFDLENBQUM7SUFDdEMsSUFBSSxDQUFDLE1BQU0sQ0FDUCxRQUFRLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxJQUFJLFFBQVEsQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQ3BDLEdBQUcsRUFBRSxDQUFDLDJDQUEyQyxRQUFRLEVBQUUsQ0FBQyxDQUFDO0lBQ2pFLElBQUksQ0FBQyxNQUFNLENBQ1AsTUFBTSxLQUFLLFVBQVUsSUFBSSxNQUFNLEtBQUssU0FBUyxFQUM3QyxHQUFHLEVBQUUsQ0FBQywrQ0FBK0MsTUFBTSxFQUFFLENBQUMsQ0FBQztJQUVuRSxNQUFNLE1BQU0sR0FDYyxFQUFDLEtBQUssRUFBRSxNQUFNLEVBQUUsS0FBSyxFQUFFLE1BQU0sRUFBRSxNQUFNLEVBQUUsT0FBTyxFQUFDLENBQUM7SUFDMUUsTUFBTSxLQUFLLEdBQXVCLEVBQUMsTUFBTSxFQUFFLGtCQUFrQixFQUFFLFFBQVEsRUFBQyxDQUFDO0lBQ3pFLE1BQU0sR0FBRyxHQUFHLE1BQU0sQ0FBQyxTQUFTLENBQ3hCLGFBQWEsRUFBRSxNQUFtQyxFQUNsRCxLQUFnQyxDQUFDLENBQUM7SUFDdEMsT0FBTyxHQUFlLENBQUM7QUFDekIsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLGFBQWEsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsY0FBYyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtFTkdJTkV9IGZyb20gJy4uLy4uL2VuZ2luZSc7XG5pbXBvcnQge0Nyb3BBbmRSZXNpemUsIENyb3BBbmRSZXNpemVBdHRycywgQ3JvcEFuZFJlc2l6ZUlucHV0c30gZnJvbSAnLi4vLi4va2VybmVsX25hbWVzJztcbmltcG9ydCB7TmFtZWRBdHRyTWFwfSBmcm9tICcuLi8uLi9rZXJuZWxfcmVnaXN0cnknO1xuaW1wb3J0IHtUZW5zb3IxRCwgVGVuc29yMkQsIFRlbnNvcjREfSBmcm9tICcuLi8uLi90ZW5zb3InO1xuaW1wb3J0IHtOYW1lZFRlbnNvck1hcH0gZnJvbSAnLi4vLi4vdGVuc29yX3R5cGVzJztcbmltcG9ydCB7Y29udmVydFRvVGVuc29yfSBmcm9tICcuLi8uLi90ZW5zb3JfdXRpbF9lbnYnO1xuaW1wb3J0IHtUZW5zb3JMaWtlfSBmcm9tICcuLi8uLi90eXBlcyc7XG5pbXBvcnQgKiBhcyB1dGlsIGZyb20gJy4uLy4uL3V0aWwnO1xuXG5pbXBvcnQge29wfSBmcm9tICcuLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIEV4dHJhY3RzIGNyb3BzIGZyb20gdGhlIGlucHV0IGltYWdlIHRlbnNvciBhbmQgcmVzaXplcyB0aGVtIHVzaW5nIGJpbGluZWFyXG4gKiBzYW1wbGluZyBvciBuZWFyZXN0IG5laWdoYm9yIHNhbXBsaW5nIChwb3NzaWJseSB3aXRoIGFzcGVjdCByYXRpbyBjaGFuZ2UpXG4gKiB0byBhIGNvbW1vbiBvdXRwdXQgc2l6ZSBzcGVjaWZpZWQgYnkgY3JvcFNpemUuXG4gKlxuICogQHBhcmFtIGltYWdlIDRkIHRlbnNvciBvZiBzaGFwZSBgW2JhdGNoLGltYWdlSGVpZ2h0LGltYWdlV2lkdGgsIGRlcHRoXWAsXG4gKiAgICAgd2hlcmUgaW1hZ2VIZWlnaHQgYW5kIGltYWdlV2lkdGggbXVzdCBiZSBwb3NpdGl2ZSwgc3BlY2lmeWluZyB0aGVcbiAqICAgICBiYXRjaCBvZiBpbWFnZXMgZnJvbSB3aGljaCB0byB0YWtlIGNyb3BzXG4gKiBAcGFyYW0gYm94ZXMgMmQgZmxvYXQzMiB0ZW5zb3Igb2Ygc2hhcGUgYFtudW1Cb3hlcywgNF1gLiBFYWNoIGVudHJ5IGlzXG4gKiAgICAgYFt5MSwgeDEsIHkyLCB4Ml1gLCB3aGVyZSBgKHkxLCB4MSlgIGFuZCBgKHkyLCB4MilgIGFyZSB0aGUgbm9ybWFsaXplZFxuICogICAgIGNvb3JkaW5hdGVzIG9mIHRoZSBib3ggaW4gdGhlIGBib3hJbmRbaV1gdGggaW1hZ2UgaW4gdGhlIGJhdGNoXG4gKiBAcGFyYW0gYm94SW5kIDFkIGludDMyIHRlbnNvciBvZiBzaGFwZSBgW251bUJveGVzXWAgd2l0aCB2YWx1ZXMgaW4gcmFuZ2VcbiAqICAgICBgWzAsIGJhdGNoKWAgdGhhdCBzcGVjaWZpZXMgdGhlIGltYWdlIHRoYXQgdGhlIGBpYC10aCBib3ggcmVmZXJzIHRvLlxuICogQHBhcmFtIGNyb3BTaXplIDFkIGludDMyIHRlbnNvciBvZiAyIGVsZW1lbnRzIGBbY3JvcEhlaWdoLCBjcm9wV2lkdGhdYFxuICogICAgIHNwZWNpZnlpbmcgdGhlIHNpemUgdG8gd2hpY2ggYWxsIGNyb3BzIGFyZSByZXNpemVkIHRvLlxuICogQHBhcmFtIG1ldGhvZCBPcHRpb25hbCBzdHJpbmcgZnJvbSBgJ2JpbGluZWFyJyB8ICduZWFyZXN0J2AsXG4gKiAgICAgZGVmYXVsdHMgdG8gYmlsaW5lYXIsIHdoaWNoIHNwZWNpZmllcyB0aGUgc2FtcGxpbmcgbWV0aG9kIGZvciByZXNpemluZ1xuICogQHBhcmFtIGV4dHJhcG9sYXRpb25WYWx1ZSBBIHRocmVzaG9sZCBmb3IgZGVjaWRpbmcgd2hlbiB0byByZW1vdmUgYm94ZXMgYmFzZWRcbiAqICAgICBvbiBzY29yZS4gRGVmYXVsdHMgdG8gMC5cbiAqIEByZXR1cm4gQSA0RCB0ZW5zb3Igb2YgdGhlIHNoYXBlIGBbbnVtQm94ZXMsY3JvcEhlaWdodCxjcm9wV2lkdGgsZGVwdGhdYFxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ0ltYWdlcycsIG5hbWVzcGFjZTogJ2ltYWdlJ31cbiAqL1xuZnVuY3Rpb24gY3JvcEFuZFJlc2l6ZV8oXG4gICAgaW1hZ2U6IFRlbnNvcjREfFRlbnNvckxpa2UsXG4gICAgYm94ZXM6IFRlbnNvcjJEfFRlbnNvckxpa2UsXG4gICAgYm94SW5kOiBUZW5zb3IxRHxUZW5zb3JMaWtlLFxuICAgIGNyb3BTaXplOiBbbnVtYmVyLCBudW1iZXJdLFxuICAgIG1ldGhvZDogJ2JpbGluZWFyJ3wnbmVhcmVzdCcgPSAnYmlsaW5lYXInLFxuICAgIGV4dHJhcG9sYXRpb25WYWx1ZSA9IDAsXG4gICAgKTogVGVuc29yNEQge1xuICBjb25zdCAkaW1hZ2UgPSBjb252ZXJ0VG9UZW5zb3IoaW1hZ2UsICdpbWFnZScsICdjcm9wQW5kUmVzaXplJyk7XG4gIGNvbnN0ICRib3hlcyA9IGNvbnZlcnRUb1RlbnNvcihib3hlcywgJ2JveGVzJywgJ2Nyb3BBbmRSZXNpemUnLCAnZmxvYXQzMicpO1xuICBjb25zdCAkYm94SW5kID0gY29udmVydFRvVGVuc29yKGJveEluZCwgJ2JveEluZCcsICdjcm9wQW5kUmVzaXplJywgJ2ludDMyJyk7XG5cbiAgY29uc3QgbnVtQm94ZXMgPSAkYm94ZXMuc2hhcGVbMF07XG5cbiAgdXRpbC5hc3NlcnQoXG4gICAgICAkaW1hZ2UucmFuayA9PT0gNCxcbiAgICAgICgpID0+ICdFcnJvciBpbiBjcm9wQW5kUmVzaXplOiBpbWFnZSBtdXN0IGJlIHJhbmsgNCwnICtcbiAgICAgICAgICBgYnV0IGdvdCByYW5rICR7JGltYWdlLnJhbmt9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRib3hlcy5yYW5rID09PSAyICYmICRib3hlcy5zaGFwZVsxXSA9PT0gNCxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjcm9wQW5kUmVzaXplOiBib3hlcyBtdXN0IGJlIGhhdmUgc2l6ZSBbJHtudW1Cb3hlc30sNF0gYCArXG4gICAgICAgICAgYGJ1dCBoYWQgc2hhcGUgJHskYm94ZXMuc2hhcGV9LmApO1xuICB1dGlsLmFzc2VydChcbiAgICAgICRib3hJbmQucmFuayA9PT0gMSAmJiAkYm94SW5kLnNoYXBlWzBdID09PSBudW1Cb3hlcyxcbiAgICAgICgpID0+IGBFcnJvciBpbiBjcm9wQW5kUmVzaXplOiBib3hJbmQgbXVzdCBiZSBoYXZlIHNpemUgWyR7bnVtQm94ZXN9XSBgICtcbiAgICAgICAgICBgYnV0IGhhZCBzaGFwZSAkeyRib3hlcy5zaGFwZX0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY3JvcFNpemUubGVuZ3RoID09PSAyLFxuICAgICAgKCkgPT4gYEVycm9yIGluIGNyb3BBbmRSZXNpemU6IGNyb3BTaXplIG11c3QgYmUgb2YgbGVuZ3RoIDIsIGJ1dCBnb3QgYCArXG4gICAgICAgICAgYGxlbmd0aCAke2Nyb3BTaXplLmxlbmd0aH0uYCk7XG4gIHV0aWwuYXNzZXJ0KFxuICAgICAgY3JvcFNpemVbMF0gPj0gMSAmJiBjcm9wU2l6ZVsxXSA+PSAxLFxuICAgICAgKCkgPT4gYGNyb3BTaXplIG11c3QgYmUgYXRsZWFzdCBbMSwxXSwgYnV0IHdhcyAke2Nyb3BTaXplfWApO1xuICB1dGlsLmFzc2VydChcbiAgICAgIG1ldGhvZCA9PT0gJ2JpbGluZWFyJyB8fCBtZXRob2QgPT09ICduZWFyZXN0JyxcbiAgICAgICgpID0+IGBtZXRob2QgbXVzdCBiZSBiaWxpbmVhciBvciBuZWFyZXN0LCBidXQgd2FzICR7bWV0aG9kfWApO1xuXG4gIGNvbnN0IGlucHV0czpcbiAgICAgIENyb3BBbmRSZXNpemVJbnB1dHMgPSB7aW1hZ2U6ICRpbWFnZSwgYm94ZXM6ICRib3hlcywgYm94SW5kOiAkYm94SW5kfTtcbiAgY29uc3QgYXR0cnM6IENyb3BBbmRSZXNpemVBdHRycyA9IHttZXRob2QsIGV4dHJhcG9sYXRpb25WYWx1ZSwgY3JvcFNpemV9O1xuICBjb25zdCByZXMgPSBFTkdJTkUucnVuS2VybmVsKFxuICAgICAgQ3JvcEFuZFJlc2l6ZSwgaW5wdXRzIGFzIHVua25vd24gYXMgTmFtZWRUZW5zb3JNYXAsXG4gICAgICBhdHRycyBhcyB1bmtub3duIGFzIE5hbWVkQXR0ck1hcCk7XG4gIHJldHVybiByZXMgYXMgVGVuc29yNEQ7XG59XG5cbmV4cG9ydCBjb25zdCBjcm9wQW5kUmVzaXplID0gLyogQF9fUFVSRV9fICovIG9wKHtjcm9wQW5kUmVzaXplX30pO1xuIl19