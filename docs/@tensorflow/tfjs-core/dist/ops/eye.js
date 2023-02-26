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
import { buffer } from './buffer';
import { expandDims } from './expand_dims';
import { op } from './operation';
import { reshape } from './reshape';
import { tile } from './tile';
/**
 * Create an identity matrix.
 *
 * @param numRows Number of rows.
 * @param numColumns Number of columns. Defaults to `numRows`.
 * @param batchShape If provided, will add the batch shape to the beginning
 *   of the shape of the returned `tf.Tensor` by repeating the identity
 *   matrix.
 * @param dtype Data type.
 * @returns Identity matrix of the specified size and data type, possibly
 *   with batch repetition if `batchShape` is specified.
 *
 * @doc {heading: 'Tensors', subheading: 'Creation'}
 */
function eye_(numRows, numColumns, batchShape, dtype = 'float32') {
    if (numColumns == null) {
        numColumns = numRows;
    }
    const buff = buffer([numRows, numColumns], dtype);
    const n = numRows <= numColumns ? numRows : numColumns;
    for (let i = 0; i < n; ++i) {
        buff.set(1, i, i);
    }
    const out = reshape(buff.toTensor(), [numRows, numColumns]);
    if (batchShape == null) {
        return out;
    }
    else {
        if (batchShape.length === 1) {
            return tile(expandDims(out, 0), [batchShape[0], 1, 1]);
        }
        else if (batchShape.length === 2) {
            // tslint:disable-next-line:no-unnecessary-type-assertion
            return tile(expandDims(expandDims(out, 0), 0), [batchShape[0], batchShape[1], 1, 1]);
        }
        else if (batchShape.length === 3) {
            // tslint:disable-next-line:no-unnecessary-type-assertion
            return tile(expandDims(expandDims(expandDims(out, 0), 0), 0), [
                batchShape[0], batchShape[1], batchShape[2], 1, 1
            ]);
        }
        else {
            throw new Error(`eye() currently supports only 1D and 2D ` +
                // tslint:disable-next-line:no-any
                `batchShapes, but received ${batchShape.length}D.`);
        }
    }
}
export const eye = /* @__PURE__ */ op({ eye_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZXllLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvZXllLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7Ozs7Ozs7Ozs7R0FlRztBQUtILE9BQU8sRUFBQyxNQUFNLEVBQUMsTUFBTSxVQUFVLENBQUM7QUFDaEMsT0FBTyxFQUFDLFVBQVUsRUFBQyxNQUFNLGVBQWUsQ0FBQztBQUN6QyxPQUFPLEVBQUMsRUFBRSxFQUFDLE1BQU0sYUFBYSxDQUFDO0FBQy9CLE9BQU8sRUFBQyxPQUFPLEVBQUMsTUFBTSxXQUFXLENBQUM7QUFDbEMsT0FBTyxFQUFDLElBQUksRUFBQyxNQUFNLFFBQVEsQ0FBQztBQUU1Qjs7Ozs7Ozs7Ozs7OztHQWFHO0FBQ0gsU0FBUyxJQUFJLENBQ1QsT0FBZSxFQUFFLFVBQW1CLEVBQ3BDLFVBSXdFLEVBQ3hFLFFBQWtCLFNBQVM7SUFDN0IsSUFBSSxVQUFVLElBQUksSUFBSSxFQUFFO1FBQ3RCLFVBQVUsR0FBRyxPQUFPLENBQUM7S0FDdEI7SUFDRCxNQUFNLElBQUksR0FBRyxNQUFNLENBQUMsQ0FBQyxPQUFPLEVBQUUsVUFBVSxDQUFDLEVBQUUsS0FBSyxDQUFDLENBQUM7SUFDbEQsTUFBTSxDQUFDLEdBQUcsT0FBTyxJQUFJLFVBQVUsQ0FBQyxDQUFDLENBQUMsT0FBTyxDQUFDLENBQUMsQ0FBQyxVQUFVLENBQUM7SUFDdkQsS0FBSyxJQUFJLENBQUMsR0FBRyxDQUFDLEVBQUUsQ0FBQyxHQUFHLENBQUMsRUFBRSxFQUFFLENBQUMsRUFBRTtRQUMxQixJQUFJLENBQUMsR0FBRyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUM7S0FDbkI7SUFDRCxNQUFNLEdBQUcsR0FBYSxPQUFPLENBQUMsSUFBSSxDQUFDLFFBQVEsRUFBRSxFQUFFLENBQUMsT0FBTyxFQUFFLFVBQVUsQ0FBQyxDQUFDLENBQUM7SUFDdEUsSUFBSSxVQUFVLElBQUksSUFBSSxFQUFFO1FBQ3RCLE9BQU8sR0FBRyxDQUFDO0tBQ1o7U0FBTTtRQUNMLElBQUksVUFBVSxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7WUFDM0IsT0FBTyxJQUFJLENBQUMsVUFBVSxDQUFDLEdBQUcsRUFBRSxDQUFDLENBQUMsRUFBRSxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQWEsQ0FBQztTQUNwRTthQUFNLElBQUksVUFBVSxDQUFDLE1BQU0sS0FBSyxDQUFDLEVBQUU7WUFDbEMseURBQXlEO1lBQ3pELE9BQU8sSUFBSSxDQUNBLFVBQVUsQ0FBQyxVQUFVLENBQUMsR0FBRyxFQUFFLENBQUMsQ0FBQyxFQUFFLENBQUMsQ0FBQyxFQUNqQyxDQUFDLFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFhLENBQUM7U0FDOUQ7YUFBTSxJQUFJLFVBQVUsQ0FBQyxNQUFNLEtBQUssQ0FBQyxFQUFFO1lBQ2xDLHlEQUF5RDtZQUN6RCxPQUFPLElBQUksQ0FBQyxVQUFVLENBQUMsVUFBVSxDQUFDLFVBQVUsQ0FBQyxHQUFHLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUUsQ0FBQyxDQUFDLEVBQUU7Z0JBQ3JELFVBQVUsQ0FBQyxDQUFDLENBQUMsRUFBRSxVQUFVLENBQUMsQ0FBQyxDQUFDLEVBQUUsVUFBVSxDQUFDLENBQUMsQ0FBQyxFQUFFLENBQUMsRUFBRSxDQUFDO2FBQ2xELENBQWEsQ0FBQztTQUN2QjthQUFNO1lBQ0wsTUFBTSxJQUFJLEtBQUssQ0FDWCwwQ0FBMEM7Z0JBQzFDLGtDQUFrQztnQkFDbEMsNkJBQThCLFVBQWtCLENBQUMsTUFBTSxJQUFJLENBQUMsQ0FBQztTQUNsRTtLQUNGO0FBQ0gsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLEdBQUcsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsSUFBSSxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cblxuaW1wb3J0IHtUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7RGF0YVR5cGV9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtidWZmZXJ9IGZyb20gJy4vYnVmZmVyJztcbmltcG9ydCB7ZXhwYW5kRGltc30gZnJvbSAnLi9leHBhbmRfZGltcyc7XG5pbXBvcnQge29wfSBmcm9tICcuL29wZXJhdGlvbic7XG5pbXBvcnQge3Jlc2hhcGV9IGZyb20gJy4vcmVzaGFwZSc7XG5pbXBvcnQge3RpbGV9IGZyb20gJy4vdGlsZSc7XG5cbi8qKlxuICogQ3JlYXRlIGFuIGlkZW50aXR5IG1hdHJpeC5cbiAqXG4gKiBAcGFyYW0gbnVtUm93cyBOdW1iZXIgb2Ygcm93cy5cbiAqIEBwYXJhbSBudW1Db2x1bW5zIE51bWJlciBvZiBjb2x1bW5zLiBEZWZhdWx0cyB0byBgbnVtUm93c2AuXG4gKiBAcGFyYW0gYmF0Y2hTaGFwZSBJZiBwcm92aWRlZCwgd2lsbCBhZGQgdGhlIGJhdGNoIHNoYXBlIHRvIHRoZSBiZWdpbm5pbmdcbiAqICAgb2YgdGhlIHNoYXBlIG9mIHRoZSByZXR1cm5lZCBgdGYuVGVuc29yYCBieSByZXBlYXRpbmcgdGhlIGlkZW50aXR5XG4gKiAgIG1hdHJpeC5cbiAqIEBwYXJhbSBkdHlwZSBEYXRhIHR5cGUuXG4gKiBAcmV0dXJucyBJZGVudGl0eSBtYXRyaXggb2YgdGhlIHNwZWNpZmllZCBzaXplIGFuZCBkYXRhIHR5cGUsIHBvc3NpYmx5XG4gKiAgIHdpdGggYmF0Y2ggcmVwZXRpdGlvbiBpZiBgYmF0Y2hTaGFwZWAgaXMgc3BlY2lmaWVkLlxuICpcbiAqIEBkb2Mge2hlYWRpbmc6ICdUZW5zb3JzJywgc3ViaGVhZGluZzogJ0NyZWF0aW9uJ31cbiAqL1xuZnVuY3Rpb24gZXllXyhcbiAgICBudW1Sb3dzOiBudW1iZXIsIG51bUNvbHVtbnM/OiBudW1iZXIsXG4gICAgYmF0Y2hTaGFwZT86XG4gICAgICAgIFtcbiAgICAgICAgICBudW1iZXJcbiAgICAgICAgXXxbbnVtYmVyLFxuICAgICAgICAgICBudW1iZXJdfFtudW1iZXIsIG51bWJlciwgbnVtYmVyXXxbbnVtYmVyLCBudW1iZXIsIG51bWJlciwgbnVtYmVyXSxcbiAgICBkdHlwZTogRGF0YVR5cGUgPSAnZmxvYXQzMicpOiBUZW5zb3IyRCB7XG4gIGlmIChudW1Db2x1bW5zID09IG51bGwpIHtcbiAgICBudW1Db2x1bW5zID0gbnVtUm93cztcbiAgfVxuICBjb25zdCBidWZmID0gYnVmZmVyKFtudW1Sb3dzLCBudW1Db2x1bW5zXSwgZHR5cGUpO1xuICBjb25zdCBuID0gbnVtUm93cyA8PSBudW1Db2x1bW5zID8gbnVtUm93cyA6IG51bUNvbHVtbnM7XG4gIGZvciAobGV0IGkgPSAwOyBpIDwgbjsgKytpKSB7XG4gICAgYnVmZi5zZXQoMSwgaSwgaSk7XG4gIH1cbiAgY29uc3Qgb3V0OiBUZW5zb3IyRCA9IHJlc2hhcGUoYnVmZi50b1RlbnNvcigpLCBbbnVtUm93cywgbnVtQ29sdW1uc10pO1xuICBpZiAoYmF0Y2hTaGFwZSA9PSBudWxsKSB7XG4gICAgcmV0dXJuIG91dDtcbiAgfSBlbHNlIHtcbiAgICBpZiAoYmF0Y2hTaGFwZS5sZW5ndGggPT09IDEpIHtcbiAgICAgIHJldHVybiB0aWxlKGV4cGFuZERpbXMob3V0LCAwKSwgW2JhdGNoU2hhcGVbMF0sIDEsIDFdKSBhcyBUZW5zb3IyRDtcbiAgICB9IGVsc2UgaWYgKGJhdGNoU2hhcGUubGVuZ3RoID09PSAyKSB7XG4gICAgICAvLyB0c2xpbnQ6ZGlzYWJsZS1uZXh0LWxpbmU6bm8tdW5uZWNlc3NhcnktdHlwZS1hc3NlcnRpb25cbiAgICAgIHJldHVybiB0aWxlKFxuICAgICAgICAgICAgICAgICBleHBhbmREaW1zKGV4cGFuZERpbXMob3V0LCAwKSwgMCksXG4gICAgICAgICAgICAgICAgIFtiYXRjaFNoYXBlWzBdLCBiYXRjaFNoYXBlWzFdLCAxLCAxXSkgYXMgVGVuc29yMkQ7XG4gICAgfSBlbHNlIGlmIChiYXRjaFNoYXBlLmxlbmd0aCA9PT0gMykge1xuICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLXVubmVjZXNzYXJ5LXR5cGUtYXNzZXJ0aW9uXG4gICAgICByZXR1cm4gdGlsZShleHBhbmREaW1zKGV4cGFuZERpbXMoZXhwYW5kRGltcyhvdXQsIDApLCAwKSwgMCksIFtcbiAgICAgICAgICAgICAgIGJhdGNoU2hhcGVbMF0sIGJhdGNoU2hhcGVbMV0sIGJhdGNoU2hhcGVbMl0sIDEsIDFcbiAgICAgICAgICAgICBdKSBhcyBUZW5zb3IyRDtcbiAgICB9IGVsc2Uge1xuICAgICAgdGhyb3cgbmV3IEVycm9yKFxuICAgICAgICAgIGBleWUoKSBjdXJyZW50bHkgc3VwcG9ydHMgb25seSAxRCBhbmQgMkQgYCArXG4gICAgICAgICAgLy8gdHNsaW50OmRpc2FibGUtbmV4dC1saW5lOm5vLWFueVxuICAgICAgICAgIGBiYXRjaFNoYXBlcywgYnV0IHJlY2VpdmVkICR7KGJhdGNoU2hhcGUgYXMgYW55KS5sZW5ndGh9RC5gKTtcbiAgICB9XG4gIH1cbn1cblxuZXhwb3J0IGNvbnN0IGV5ZSA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7ZXllX30pO1xuIl19