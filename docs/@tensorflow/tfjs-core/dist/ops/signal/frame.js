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
import { concat } from '../concat';
import { fill } from '../fill';
import { op } from '../operation';
import { reshape } from '../reshape';
import { slice } from '../slice';
import { tensor2d } from '../tensor2d';
/**
 * Expands input into frames of frameLength.
 * Slides a window size with frameStep.
 *
 * ```js
 * tf.signal.frame([1, 2, 3], 2, 1).print();
 * ```
 * @param signal The input tensor to be expanded
 * @param frameLength Length of each frame
 * @param frameStep The frame hop size in samples.
 * @param padEnd Whether to pad the end of signal with padValue.
 * @param padValue A number to use where the input signal does
 *     not exist when padEnd is True.
 *
 * @doc {heading: 'Operations', subheading: 'Signal', namespace: 'signal'}
 */
function frame_(signal, frameLength, frameStep, padEnd = false, padValue = 0) {
    let start = 0;
    const output = [];
    while (start + frameLength <= signal.size) {
        output.push(slice(signal, start, frameLength));
        start += frameStep;
    }
    if (padEnd) {
        while (start < signal.size) {
            const padLen = (start + frameLength) - signal.size;
            const pad = concat([
                slice(signal, start, frameLength - padLen), fill([padLen], padValue)
            ]);
            output.push(pad);
            start += frameStep;
        }
    }
    if (output.length === 0) {
        return tensor2d([], [0, frameLength]);
    }
    return reshape(concat(output), [output.length, frameLength]);
}
export const frame = /* @__PURE__ */ op({ frame_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZnJhbWUuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi8uLi8uLi90ZmpzLWNvcmUvc3JjL29wcy9zaWduYWwvZnJhbWUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLE1BQU0sRUFBQyxNQUFNLFdBQVcsQ0FBQztBQUNqQyxPQUFPLEVBQUMsSUFBSSxFQUFDLE1BQU0sU0FBUyxDQUFDO0FBQzdCLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxjQUFjLENBQUM7QUFDaEMsT0FBTyxFQUFDLE9BQU8sRUFBQyxNQUFNLFlBQVksQ0FBQztBQUNuQyxPQUFPLEVBQUMsS0FBSyxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxRQUFRLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFckM7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBQ0gsU0FBUyxNQUFNLENBQ1gsTUFBZ0IsRUFBRSxXQUFtQixFQUFFLFNBQWlCLEVBQUUsTUFBTSxHQUFHLEtBQUssRUFDeEUsUUFBUSxHQUFHLENBQUM7SUFDZCxJQUFJLEtBQUssR0FBRyxDQUFDLENBQUM7SUFDZCxNQUFNLE1BQU0sR0FBYSxFQUFFLENBQUM7SUFDNUIsT0FBTyxLQUFLLEdBQUcsV0FBVyxJQUFJLE1BQU0sQ0FBQyxJQUFJLEVBQUU7UUFDekMsTUFBTSxDQUFDLElBQUksQ0FBQyxLQUFLLENBQUMsTUFBTSxFQUFFLEtBQUssRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDO1FBQy9DLEtBQUssSUFBSSxTQUFTLENBQUM7S0FDcEI7SUFFRCxJQUFJLE1BQU0sRUFBRTtRQUNWLE9BQU8sS0FBSyxHQUFHLE1BQU0sQ0FBQyxJQUFJLEVBQUU7WUFDMUIsTUFBTSxNQUFNLEdBQUcsQ0FBQyxLQUFLLEdBQUcsV0FBVyxDQUFDLEdBQUcsTUFBTSxDQUFDLElBQUksQ0FBQztZQUNuRCxNQUFNLEdBQUcsR0FBRyxNQUFNLENBQUM7Z0JBQ2pCLEtBQUssQ0FBQyxNQUFNLEVBQUUsS0FBSyxFQUFFLFdBQVcsR0FBRyxNQUFNLENBQUMsRUFBRSxJQUFJLENBQUMsQ0FBQyxNQUFNLENBQUMsRUFBRSxRQUFRLENBQUM7YUFDckUsQ0FBQyxDQUFDO1lBQ0gsTUFBTSxDQUFDLElBQUksQ0FBQyxHQUFHLENBQUMsQ0FBQztZQUNqQixLQUFLLElBQUksU0FBUyxDQUFDO1NBQ3BCO0tBQ0Y7SUFFRCxJQUFJLE1BQU0sQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1FBQ3ZCLE9BQU8sUUFBUSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDO0tBQ3ZDO0lBRUQsT0FBTyxPQUFPLENBQUMsTUFBTSxDQUFDLE1BQU0sQ0FBQyxFQUFFLENBQUMsTUFBTSxDQUFDLE1BQU0sRUFBRSxXQUFXLENBQUMsQ0FBQyxDQUFDO0FBQy9ELENBQUM7QUFDRCxNQUFNLENBQUMsTUFBTSxLQUFLLEdBQUcsZUFBZSxDQUFDLEVBQUUsQ0FBQyxFQUFDLE1BQU0sRUFBQyxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOSBHb29nbGUgTExDLiBBbGwgUmlnaHRzIFJlc2VydmVkLlxuICogTGljZW5zZWQgdW5kZXIgdGhlIEFwYWNoZSBMaWNlbnNlLCBWZXJzaW9uIDIuMCAodGhlIFwiTGljZW5zZVwiKTtcbiAqIHlvdSBtYXkgbm90IHVzZSB0aGlzIGZpbGUgZXhjZXB0IGluIGNvbXBsaWFuY2Ugd2l0aCB0aGUgTGljZW5zZS5cbiAqIFlvdSBtYXkgb2J0YWluIGEgY29weSBvZiB0aGUgTGljZW5zZSBhdFxuICpcbiAqIGh0dHA6Ly93d3cuYXBhY2hlLm9yZy9saWNlbnNlcy9MSUNFTlNFLTIuMFxuICpcbiAqIFVubGVzcyByZXF1aXJlZCBieSBhcHBsaWNhYmxlIGxhdyBvciBhZ3JlZWQgdG8gaW4gd3JpdGluZywgc29mdHdhcmVcbiAqIGRpc3RyaWJ1dGVkIHVuZGVyIHRoZSBMaWNlbnNlIGlzIGRpc3RyaWJ1dGVkIG9uIGFuIFwiQVMgSVNcIiBCQVNJUyxcbiAqIFdJVEhPVVQgV0FSUkFOVElFUyBPUiBDT05ESVRJT05TIE9GIEFOWSBLSU5ELCBlaXRoZXIgZXhwcmVzcyBvciBpbXBsaWVkLlxuICogU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZFxuICogbGltaXRhdGlvbnMgdW5kZXIgdGhlIExpY2Vuc2UuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7VGVuc29yLCBUZW5zb3IxRH0gZnJvbSAnLi4vLi4vdGVuc29yJztcbmltcG9ydCB7Y29uY2F0fSBmcm9tICcuLi9jb25jYXQnO1xuaW1wb3J0IHtmaWxsfSBmcm9tICcuLi9maWxsJztcbmltcG9ydCB7b3B9IGZyb20gJy4uL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4uL3Jlc2hhcGUnO1xuaW1wb3J0IHtzbGljZX0gZnJvbSAnLi4vc2xpY2UnO1xuaW1wb3J0IHt0ZW5zb3IyZH0gZnJvbSAnLi4vdGVuc29yMmQnO1xuXG4vKipcbiAqIEV4cGFuZHMgaW5wdXQgaW50byBmcmFtZXMgb2YgZnJhbWVMZW5ndGguXG4gKiBTbGlkZXMgYSB3aW5kb3cgc2l6ZSB3aXRoIGZyYW1lU3RlcC5cbiAqXG4gKiBgYGBqc1xuICogdGYuc2lnbmFsLmZyYW1lKFsxLCAyLCAzXSwgMiwgMSkucHJpbnQoKTtcbiAqIGBgYFxuICogQHBhcmFtIHNpZ25hbCBUaGUgaW5wdXQgdGVuc29yIHRvIGJlIGV4cGFuZGVkXG4gKiBAcGFyYW0gZnJhbWVMZW5ndGggTGVuZ3RoIG9mIGVhY2ggZnJhbWVcbiAqIEBwYXJhbSBmcmFtZVN0ZXAgVGhlIGZyYW1lIGhvcCBzaXplIGluIHNhbXBsZXMuXG4gKiBAcGFyYW0gcGFkRW5kIFdoZXRoZXIgdG8gcGFkIHRoZSBlbmQgb2Ygc2lnbmFsIHdpdGggcGFkVmFsdWUuXG4gKiBAcGFyYW0gcGFkVmFsdWUgQSBudW1iZXIgdG8gdXNlIHdoZXJlIHRoZSBpbnB1dCBzaWduYWwgZG9lc1xuICogICAgIG5vdCBleGlzdCB3aGVuIHBhZEVuZCBpcyBUcnVlLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdPcGVyYXRpb25zJywgc3ViaGVhZGluZzogJ1NpZ25hbCcsIG5hbWVzcGFjZTogJ3NpZ25hbCd9XG4gKi9cbmZ1bmN0aW9uIGZyYW1lXyhcbiAgICBzaWduYWw6IFRlbnNvcjFELCBmcmFtZUxlbmd0aDogbnVtYmVyLCBmcmFtZVN0ZXA6IG51bWJlciwgcGFkRW5kID0gZmFsc2UsXG4gICAgcGFkVmFsdWUgPSAwKTogVGVuc29yIHtcbiAgbGV0IHN0YXJ0ID0gMDtcbiAgY29uc3Qgb3V0cHV0OiBUZW5zb3JbXSA9IFtdO1xuICB3aGlsZSAoc3RhcnQgKyBmcmFtZUxlbmd0aCA8PSBzaWduYWwuc2l6ZSkge1xuICAgIG91dHB1dC5wdXNoKHNsaWNlKHNpZ25hbCwgc3RhcnQsIGZyYW1lTGVuZ3RoKSk7XG4gICAgc3RhcnQgKz0gZnJhbWVTdGVwO1xuICB9XG5cbiAgaWYgKHBhZEVuZCkge1xuICAgIHdoaWxlIChzdGFydCA8IHNpZ25hbC5zaXplKSB7XG4gICAgICBjb25zdCBwYWRMZW4gPSAoc3RhcnQgKyBmcmFtZUxlbmd0aCkgLSBzaWduYWwuc2l6ZTtcbiAgICAgIGNvbnN0IHBhZCA9IGNvbmNhdChbXG4gICAgICAgIHNsaWNlKHNpZ25hbCwgc3RhcnQsIGZyYW1lTGVuZ3RoIC0gcGFkTGVuKSwgZmlsbChbcGFkTGVuXSwgcGFkVmFsdWUpXG4gICAgICBdKTtcbiAgICAgIG91dHB1dC5wdXNoKHBhZCk7XG4gICAgICBzdGFydCArPSBmcmFtZVN0ZXA7XG4gICAgfVxuICB9XG5cbiAgaWYgKG91dHB1dC5sZW5ndGggPT09IDApIHtcbiAgICByZXR1cm4gdGVuc29yMmQoW10sIFswLCBmcmFtZUxlbmd0aF0pO1xuICB9XG5cbiAgcmV0dXJuIHJlc2hhcGUoY29uY2F0KG91dHB1dCksIFtvdXRwdXQubGVuZ3RoLCBmcmFtZUxlbmd0aF0pO1xufVxuZXhwb3J0IGNvbnN0IGZyYW1lID0gLyogQF9fUFVSRV9fICovIG9wKHtmcmFtZV99KTtcbiJdfQ==