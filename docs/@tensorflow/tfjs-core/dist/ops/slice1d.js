/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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
import { convertToTensor } from '../tensor_util_env';
import * as util from '../util';
import { op } from './operation';
import { slice } from './slice';
/**
 * Extracts a 1D slice from 1D array starting at coordinates `begin` and is
 * of length `size`. See `slice` for details.
 */
function slice1d_(x, begin, size) {
    const $x = convertToTensor(x, 'x', 'slice1d');
    util.assert($x.rank === 1, () => `slice1d expects a rank-1 tensor, but got a rank-${$x.rank} tensor`);
    return slice($x, [begin], [size]);
}
export const slice1d = /* @__PURE__ */ op({ slice1d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoic2xpY2UxZC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uLy4uL3RmanMtY29yZS9zcmMvb3BzL3NsaWNlMWQudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUE7Ozs7Ozs7Ozs7Ozs7OztHQWVHO0FBR0gsT0FBTyxFQUFDLGVBQWUsRUFBQyxNQUFNLG9CQUFvQixDQUFDO0FBRW5ELE9BQU8sS0FBSyxJQUFJLE1BQU0sU0FBUyxDQUFDO0FBRWhDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFDL0IsT0FBTyxFQUFDLEtBQUssRUFBQyxNQUFNLFNBQVMsQ0FBQztBQUU5Qjs7O0dBR0c7QUFDSCxTQUFTLFFBQVEsQ0FDYixDQUFzQixFQUFFLEtBQWEsRUFBRSxJQUFZO0lBQ3JELE1BQU0sRUFBRSxHQUFHLGVBQWUsQ0FBQyxDQUFDLEVBQUUsR0FBRyxFQUFFLFNBQVMsQ0FBQyxDQUFDO0lBQzlDLElBQUksQ0FBQyxNQUFNLENBQ1AsRUFBRSxDQUFDLElBQUksS0FBSyxDQUFDLEVBQ2IsR0FBRyxFQUFFLENBQ0QsbURBQW1ELEVBQUUsQ0FBQyxJQUFJLFNBQVMsQ0FBQyxDQUFDO0lBQzdFLE9BQU8sS0FBSyxDQUFDLEVBQUUsRUFBRSxDQUFDLEtBQUssQ0FBQyxFQUFFLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQztBQUNwQyxDQUFDO0FBQ0QsTUFBTSxDQUFDLE1BQU0sT0FBTyxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxRQUFRLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMTggR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG5pbXBvcnQge1RlbnNvcjFEfSBmcm9tICcuLi90ZW5zb3InO1xuaW1wb3J0IHtjb252ZXJ0VG9UZW5zb3J9IGZyb20gJy4uL3RlbnNvcl91dGlsX2Vudic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcbmltcG9ydCAqIGFzIHV0aWwgZnJvbSAnLi4vdXRpbCc7XG5cbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcbmltcG9ydCB7c2xpY2V9IGZyb20gJy4vc2xpY2UnO1xuXG4vKipcbiAqIEV4dHJhY3RzIGEgMUQgc2xpY2UgZnJvbSAxRCBhcnJheSBzdGFydGluZyBhdCBjb29yZGluYXRlcyBgYmVnaW5gIGFuZCBpc1xuICogb2YgbGVuZ3RoIGBzaXplYC4gU2VlIGBzbGljZWAgZm9yIGRldGFpbHMuXG4gKi9cbmZ1bmN0aW9uIHNsaWNlMWRfKFxuICAgIHg6IFRlbnNvcjFEfFRlbnNvckxpa2UsIGJlZ2luOiBudW1iZXIsIHNpemU6IG51bWJlcik6IFRlbnNvcjFEIHtcbiAgY29uc3QgJHggPSBjb252ZXJ0VG9UZW5zb3IoeCwgJ3gnLCAnc2xpY2UxZCcpO1xuICB1dGlsLmFzc2VydChcbiAgICAgICR4LnJhbmsgPT09IDEsXG4gICAgICAoKSA9PlxuICAgICAgICAgIGBzbGljZTFkIGV4cGVjdHMgYSByYW5rLTEgdGVuc29yLCBidXQgZ290IGEgcmFuay0keyR4LnJhbmt9IHRlbnNvcmApO1xuICByZXR1cm4gc2xpY2UoJHgsIFtiZWdpbl0sIFtzaXplXSk7XG59XG5leHBvcnQgY29uc3Qgc2xpY2UxZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7c2xpY2UxZF99KTtcbiJdfQ==