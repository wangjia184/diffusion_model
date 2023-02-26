import { concat } from './concat';
import { op } from './operation';
/**
 * Concatenates a list of `tf.Tensor3D`s along an axis.
 * See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *
 * B: shape(2, 1, 3) = | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * C = tf.concat3d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 1, 3) = | r1, g1, b1 |
 *                     | r2, g2, b2 |
 *                     | r3, g3, b3 |
 *                     | r4, g4, b4 |
 *
 * if axis = 1:
 * C: shape(2, 2, 3) = | r1, g1, b1, r3, g3, b3 |
 *                     | r2, g2, b2, r4, g4, b4 |
 *
 * if axis = 2:
 * C = shape(2, 1, 6) = | r1, g1, b1, r3, g3, b3 |
 *                      | r2, g2, b2, r4, g4, b4 |
 *
 * @param tensors A list of`tf.Tensor`s to concatenate.
 * @param axis The axis to concate along.
 * @return The concatenated array.
 */
function concat3d_(tensors, axis) {
    return concat(tensors, axis);
}
export const concat3d = /* @__PURE__ */ op({ concat3d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uY2F0XzNkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29uY2F0XzNkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQW1CQSxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7OztHQThCRztBQUNILFNBQVMsU0FBUyxDQUNkLE9BQW1DLEVBQUUsSUFBWTtJQUNuRCxPQUFPLE1BQU0sQ0FBQyxPQUFPLEVBQUUsSUFBSSxDQUFDLENBQUM7QUFDL0IsQ0FBQztBQUVELE1BQU0sQ0FBQyxNQUFNLFFBQVEsR0FBRyxlQUFlLENBQUMsRUFBRSxDQUFDLEVBQUMsU0FBUyxFQUFDLENBQUMsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDIwIEdvb2dsZSBMTEMuIEFsbCBSaWdodHMgUmVzZXJ2ZWQuXG4gKiBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgXCJMaWNlbnNlXCIpO1xuICogeW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLlxuICogWW91IG1heSBvYnRhaW4gYSBjb3B5IG9mIHRoZSBMaWNlbnNlIGF0XG4gKlxuICogaHR0cDovL3d3dy5hcGFjaGUub3JnL2xpY2Vuc2VzL0xJQ0VOU0UtMi4wXG4gKlxuICogVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZVxuICogZGlzdHJpYnV0ZWQgdW5kZXIgdGhlIExpY2Vuc2UgaXMgZGlzdHJpYnV0ZWQgb24gYW4gXCJBUyBJU1wiIEJBU0lTLFxuICogV0lUSE9VVCBXQVJSQU5USUVTIE9SIENPTkRJVElPTlMgT0YgQU5ZIEtJTkQsIGVpdGhlciBleHByZXNzIG9yIGltcGxpZWQuXG4gKiBTZWUgdGhlIExpY2Vuc2UgZm9yIHRoZSBzcGVjaWZpYyBsYW5ndWFnZSBnb3Zlcm5pbmcgcGVybWlzc2lvbnMgYW5kXG4gKiBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS5cbiAqID09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09XG4gKi9cbmltcG9ydCB7VGVuc29yM0R9IGZyb20gJy4uL3RlbnNvcic7XG5pbXBvcnQge1RlbnNvckxpa2V9IGZyb20gJy4uL3R5cGVzJztcblxuaW1wb3J0IHtjb25jYXR9IGZyb20gJy4vY29uY2F0JztcbmltcG9ydCB7b3B9IGZyb20gJy4vb3BlcmF0aW9uJztcblxuLyoqXG4gKiBDb25jYXRlbmF0ZXMgYSBsaXN0IG9mIGB0Zi5UZW5zb3IzRGBzIGFsb25nIGFuIGF4aXMuXG4gKiBTZWUgYGNvbmNhdGAgZm9yIGRldGFpbHMuXG4gKlxuICogRm9yIGV4YW1wbGUsIGlmOlxuICogQTogc2hhcGUoMiwgMSwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICpcbiAqIEI6IHNoYXBlKDIsIDEsIDMpID0gfCByMywgZzMsIGIzIHxcbiAqICAgICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAqXG4gKiBDID0gdGYuY29uY2F0M2QoW0EsIEJdLCBheGlzKVxuICpcbiAqIGlmIGF4aXMgPSAwOlxuICogQzogc2hhcGUoNCwgMSwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICogICAgICAgICAgICAgICAgICAgICB8IHIzLCBnMywgYjMgfFxuICogICAgICAgICAgICAgICAgICAgICB8IHI0LCBnNCwgYjQgfFxuICpcbiAqIGlmIGF4aXMgPSAxOlxuICogQzogc2hhcGUoMiwgMiwgMykgPSB8IHIxLCBnMSwgYjEsIHIzLCBnMywgYjMgfFxuICogICAgICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIsIHI0LCBnNCwgYjQgfFxuICpcbiAqIGlmIGF4aXMgPSAyOlxuICogQyA9IHNoYXBlKDIsIDEsIDYpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAqICAgICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiwgcjQsIGc0LCBiNCB8XG4gKlxuICogQHBhcmFtIHRlbnNvcnMgQSBsaXN0IG9mYHRmLlRlbnNvcmBzIHRvIGNvbmNhdGVuYXRlLlxuICogQHBhcmFtIGF4aXMgVGhlIGF4aXMgdG8gY29uY2F0ZSBhbG9uZy5cbiAqIEByZXR1cm4gVGhlIGNvbmNhdGVuYXRlZCBhcnJheS5cbiAqL1xuZnVuY3Rpb24gY29uY2F0M2RfKFxuICAgIHRlbnNvcnM6IEFycmF5PFRlbnNvcjNEfFRlbnNvckxpa2U+LCBheGlzOiBudW1iZXIpOiBUZW5zb3IzRCB7XG4gIHJldHVybiBjb25jYXQodGVuc29ycywgYXhpcyk7XG59XG5cbmV4cG9ydCBjb25zdCBjb25jYXQzZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7Y29uY2F0M2RffSk7XG4iXX0=