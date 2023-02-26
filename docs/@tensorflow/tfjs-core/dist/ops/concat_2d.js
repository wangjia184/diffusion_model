import { concat } from './concat';
import { op } from './operation';
/**
 * Concatenates a list of`tf.Tensor2D`s along an axis. See `concat` for details.
 *
 * For example, if:
 * A: shape(2, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *
 * B: shape(2, 3) = | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * C = tf.concat2d([A, B], axis)
 *
 * if axis = 0:
 * C: shape(4, 3) = | r1, g1, b1 |
 *                  | r2, g2, b2 |
 *                  | r3, g3, b3 |
 *                  | r4, g4, b4 |
 *
 * if axis = 1:
 * C = shape(2, 6) = | r1, g1, b1, r3, g3, b3 |
 *                   | r2, g2, b2, r4, g4, b4 |
 *
 *
 * @param tensors A list of `tf.Tensor`s to concatenate.
 * @param axis The axis to concatenate along.
 * @return The concatenated array.
 */
function concat2d_(tensors, axis) {
    return concat(tensors, axis);
}
export const concat2d = /* @__PURE__ */ op({ concat2d_ });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29uY2F0XzJkLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1jb3JlL3NyYy9vcHMvY29uY2F0XzJkLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQW1CQSxPQUFPLEVBQUMsTUFBTSxFQUFDLE1BQU0sVUFBVSxDQUFDO0FBQ2hDLE9BQU8sRUFBQyxFQUFFLEVBQUMsTUFBTSxhQUFhLENBQUM7QUFFL0I7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0dBMEJHO0FBQ0gsU0FBUyxTQUFTLENBQ2QsT0FBbUMsRUFBRSxJQUFZO0lBQ25ELE9BQU8sTUFBTSxDQUFDLE9BQU8sRUFBRSxJQUFJLENBQUMsQ0FBQztBQUMvQixDQUFDO0FBRUQsTUFBTSxDQUFDLE1BQU0sUUFBUSxHQUFHLGVBQWUsQ0FBQyxFQUFFLENBQUMsRUFBQyxTQUFTLEVBQUMsQ0FBQyxDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiLyoqXG4gKiBAbGljZW5zZVxuICogQ29weXJpZ2h0IDIwMjAgR29vZ2xlIExMQy4gQWxsIFJpZ2h0cyBSZXNlcnZlZC5cbiAqIExpY2Vuc2VkIHVuZGVyIHRoZSBBcGFjaGUgTGljZW5zZSwgVmVyc2lvbiAyLjAgKHRoZSBcIkxpY2Vuc2VcIik7XG4gKiB5b3UgbWF5IG5vdCB1c2UgdGhpcyBmaWxlIGV4Y2VwdCBpbiBjb21wbGlhbmNlIHdpdGggdGhlIExpY2Vuc2UuXG4gKiBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXRcbiAqXG4gKiBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjBcbiAqXG4gKiBVbmxlc3MgcmVxdWlyZWQgYnkgYXBwbGljYWJsZSBsYXcgb3IgYWdyZWVkIHRvIGluIHdyaXRpbmcsIHNvZnR3YXJlXG4gKiBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiBcIkFTIElTXCIgQkFTSVMsXG4gKiBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC5cbiAqIFNlZSB0aGUgTGljZW5zZSBmb3IgdGhlIHNwZWNpZmljIGxhbmd1YWdlIGdvdmVybmluZyBwZXJtaXNzaW9ucyBhbmRcbiAqIGxpbWl0YXRpb25zIHVuZGVyIHRoZSBMaWNlbnNlLlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuaW1wb3J0IHtUZW5zb3IyRH0gZnJvbSAnLi4vdGVuc29yJztcbmltcG9ydCB7VGVuc29yTGlrZX0gZnJvbSAnLi4vdHlwZXMnO1xuXG5pbXBvcnQge2NvbmNhdH0gZnJvbSAnLi9jb25jYXQnO1xuaW1wb3J0IHtvcH0gZnJvbSAnLi9vcGVyYXRpb24nO1xuXG4vKipcbiAqIENvbmNhdGVuYXRlcyBhIGxpc3Qgb2ZgdGYuVGVuc29yMkRgcyBhbG9uZyBhbiBheGlzLiBTZWUgYGNvbmNhdGAgZm9yIGRldGFpbHMuXG4gKlxuICogRm9yIGV4YW1wbGUsIGlmOlxuICogQTogc2hhcGUoMiwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICogICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICpcbiAqIEI6IHNoYXBlKDIsIDMpID0gfCByMywgZzMsIGIzIHxcbiAqICAgICAgICAgICAgICAgICAgfCByNCwgZzQsIGI0IHxcbiAqXG4gKiBDID0gdGYuY29uY2F0MmQoW0EsIEJdLCBheGlzKVxuICpcbiAqIGlmIGF4aXMgPSAwOlxuICogQzogc2hhcGUoNCwgMykgPSB8IHIxLCBnMSwgYjEgfFxuICogICAgICAgICAgICAgICAgICB8IHIyLCBnMiwgYjIgfFxuICogICAgICAgICAgICAgICAgICB8IHIzLCBnMywgYjMgfFxuICogICAgICAgICAgICAgICAgICB8IHI0LCBnNCwgYjQgfFxuICpcbiAqIGlmIGF4aXMgPSAxOlxuICogQyA9IHNoYXBlKDIsIDYpID0gfCByMSwgZzEsIGIxLCByMywgZzMsIGIzIHxcbiAqICAgICAgICAgICAgICAgICAgIHwgcjIsIGcyLCBiMiwgcjQsIGc0LCBiNCB8XG4gKlxuICpcbiAqIEBwYXJhbSB0ZW5zb3JzIEEgbGlzdCBvZiBgdGYuVGVuc29yYHMgdG8gY29uY2F0ZW5hdGUuXG4gKiBAcGFyYW0gYXhpcyBUaGUgYXhpcyB0byBjb25jYXRlbmF0ZSBhbG9uZy5cbiAqIEByZXR1cm4gVGhlIGNvbmNhdGVuYXRlZCBhcnJheS5cbiAqL1xuZnVuY3Rpb24gY29uY2F0MmRfKFxuICAgIHRlbnNvcnM6IEFycmF5PFRlbnNvcjJEfFRlbnNvckxpa2U+LCBheGlzOiBudW1iZXIpOiBUZW5zb3IyRCB7XG4gIHJldHVybiBjb25jYXQodGVuc29ycywgYXhpcyk7XG59XG5cbmV4cG9ydCBjb25zdCBjb25jYXQyZCA9IC8qIEBfX1BVUkVfXyAqLyBvcCh7Y29uY2F0MmRffSk7XG4iXX0=