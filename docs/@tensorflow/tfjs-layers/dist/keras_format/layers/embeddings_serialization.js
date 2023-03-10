/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
// We can't easily extract a string[] from the string union type, but we can
// recapitulate the list, enforcing at compile time that the values are valid.
/**
 * A string array of valid EmbeddingLayer class names.
 *
 * This is guaranteed to match the `EmbeddingLayerClassName` union type.
 */
export const embeddingLayerClassNames = [
    'Embedding',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiZW1iZWRkaW5nc19zZXJpYWxpemF0aW9uLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL2tlcmFzX2Zvcm1hdC9sYXllcnMvZW1iZWRkaW5nc19zZXJpYWxpemF0aW9uLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBOzs7Ozs7OztHQVFHO0FBd0JILDRFQUE0RTtBQUM1RSw4RUFBOEU7QUFFOUU7Ozs7R0FJRztBQUNILE1BQU0sQ0FBQyxNQUFNLHdCQUF3QixHQUE4QjtJQUNqRSxXQUFXO0NBQ1osQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbIi8qKlxuICogQGxpY2Vuc2VcbiAqIENvcHlyaWdodCAyMDE4IEdvb2dsZSBMTENcbiAqXG4gKiBVc2Ugb2YgdGhpcyBzb3VyY2UgY29kZSBpcyBnb3Zlcm5lZCBieSBhbiBNSVQtc3R5bGVcbiAqIGxpY2Vuc2UgdGhhdCBjYW4gYmUgZm91bmQgaW4gdGhlIExJQ0VOU0UgZmlsZSBvciBhdFxuICogaHR0cHM6Ly9vcGVuc291cmNlLm9yZy9saWNlbnNlcy9NSVQuXG4gKiA9PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PVxuICovXG5cbmltcG9ydCB7Q29uc3RyYWludFNlcmlhbGl6YXRpb259IGZyb20gJy4uL2NvbnN0cmFpbnRfY29uZmlnJztcbmltcG9ydCB7SW5pdGlhbGl6ZXJTZXJpYWxpemF0aW9ufSBmcm9tICcuLi9pbml0aWFsaXplcl9jb25maWcnO1xuaW1wb3J0IHtSZWd1bGFyaXplclNlcmlhbGl6YXRpb259IGZyb20gJy4uL3JlZ3VsYXJpemVyX2NvbmZpZyc7XG5pbXBvcnQge0Jhc2VMYXllclNlcmlhbGl6YXRpb24sIExheWVyQ29uZmlnfSBmcm9tICcuLi90b3BvbG9neV9jb25maWcnO1xuXG5leHBvcnQgaW50ZXJmYWNlIEVtYmVkZGluZ0xheWVyQ29uZmlnIGV4dGVuZHMgTGF5ZXJDb25maWcge1xuICBpbnB1dF9kaW06IG51bWJlcjtcbiAgb3V0cHV0X2RpbTogbnVtYmVyO1xuICBlbWJlZGRpbmdzX2luaXRpYWxpemVyPzogSW5pdGlhbGl6ZXJTZXJpYWxpemF0aW9uO1xuICBlbWJlZGRpbmdzX3JlZ3VsYXJpemVyPzogUmVndWxhcml6ZXJTZXJpYWxpemF0aW9uO1xuICBhY3Rpdml0eV9yZWd1bGFyaXplcj86IFJlZ3VsYXJpemVyU2VyaWFsaXphdGlvbjtcbiAgZW1iZWRkaW5nc19jb25zdHJhaW50PzogQ29uc3RyYWludFNlcmlhbGl6YXRpb247XG4gIG1hc2tfemVybz86IGJvb2xlYW47XG4gIGlucHV0X2xlbmd0aD86IG51bWJlcnxudW1iZXJbXTtcbn1cblxuLy8gVXBkYXRlIGVtYmVkZGluZ0xheWVyQ2xhc3NOYW1lcyBiZWxvdyBpbiBjb25jZXJ0IHdpdGggdGhpcy5cbmV4cG9ydCB0eXBlIEVtYmVkZGluZ0xheWVyU2VyaWFsaXphdGlvbiA9XG4gICAgQmFzZUxheWVyU2VyaWFsaXphdGlvbjwnRW1iZWRkaW5nJywgRW1iZWRkaW5nTGF5ZXJDb25maWc+O1xuXG5leHBvcnQgdHlwZSBFbWJlZGRpbmdMYXllckNsYXNzTmFtZSA9IEVtYmVkZGluZ0xheWVyU2VyaWFsaXphdGlvblsnY2xhc3NfbmFtZSddO1xuXG4vLyBXZSBjYW4ndCBlYXNpbHkgZXh0cmFjdCBhIHN0cmluZ1tdIGZyb20gdGhlIHN0cmluZyB1bmlvbiB0eXBlLCBidXQgd2UgY2FuXG4vLyByZWNhcGl0dWxhdGUgdGhlIGxpc3QsIGVuZm9yY2luZyBhdCBjb21waWxlIHRpbWUgdGhhdCB0aGUgdmFsdWVzIGFyZSB2YWxpZC5cblxuLyoqXG4gKiBBIHN0cmluZyBhcnJheSBvZiB2YWxpZCBFbWJlZGRpbmdMYXllciBjbGFzcyBuYW1lcy5cbiAqXG4gKiBUaGlzIGlzIGd1YXJhbnRlZWQgdG8gbWF0Y2ggdGhlIGBFbWJlZGRpbmdMYXllckNsYXNzTmFtZWAgdW5pb24gdHlwZS5cbiAqL1xuZXhwb3J0IGNvbnN0IGVtYmVkZGluZ0xheWVyQ2xhc3NOYW1lczogRW1iZWRkaW5nTGF5ZXJDbGFzc05hbWVbXSA9IFtcbiAgJ0VtYmVkZGluZycsXG5dO1xuIl19