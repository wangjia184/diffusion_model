/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */
/* original source: keras/regularizers.py */
import * as tfc from '@tensorflow/tfjs-core';
import { abs, add, serialization, sum, tidy, zeros } from '@tensorflow/tfjs-core';
import * as K from './backend/tfjs_backend';
import { deserializeKerasObject, serializeKerasObject } from './utils/generic_utils';
function assertObjectArgs(args) {
    if (args != null && typeof args !== 'object') {
        throw new Error(`Argument to L1L2 regularizer's constructor is expected to be an ` +
            `object, but received: ${args}`);
    }
}
/**
 * Regularizer base class.
 */
export class Regularizer extends serialization.Serializable {
}
export class L1L2 extends Regularizer {
    constructor(args) {
        super();
        assertObjectArgs(args);
        this.l1 = args == null || args.l1 == null ? 0.01 : args.l1;
        this.l2 = args == null || args.l2 == null ? 0.01 : args.l2;
        this.hasL1 = this.l1 !== 0;
        this.hasL2 = this.l2 !== 0;
    }
    /**
     * Porting note: Renamed from __call__.
     * @param x Variable of which to calculate the regularization score.
     */
    apply(x) {
        return tidy(() => {
            let regularization = zeros([1]);
            if (this.hasL1) {
                regularization = add(regularization, sum(tfc.mul(this.l1, abs(x))));
            }
            if (this.hasL2) {
                regularization =
                    add(regularization, sum(tfc.mul(this.l2, K.square(x))));
            }
            return tfc.reshape(regularization, []);
        });
    }
    getConfig() {
        return { 'l1': this.l1, 'l2': this.l2 };
    }
    /** @nocollapse */
    static fromConfig(cls, config) {
        return new cls({ l1: config['l1'], l2: config['l2'] });
    }
}
/** @nocollapse */
L1L2.className = 'L1L2';
serialization.registerClass(L1L2);
export function l1(args) {
    assertObjectArgs(args);
    return new L1L2({ l1: args != null ? args.l1 : null, l2: 0 });
}
export function l2(args) {
    assertObjectArgs(args);
    return new L1L2({ l2: args != null ? args.l2 : null, l1: 0 });
}
// Maps the JavaScript-like identifier keys to the corresponding keras symbols.
export const REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP = {
    'l1l2': 'L1L2'
};
export function serializeRegularizer(constraint) {
    return serializeKerasObject(constraint);
}
export function deserializeRegularizer(config, customObjects = {}) {
    return deserializeKerasObject(config, serialization.SerializationMap.getMap().classNameMap, customObjects, 'regularizer');
}
export function getRegularizer(identifier) {
    if (identifier == null) {
        return null;
    }
    if (typeof identifier === 'string') {
        const className = identifier in REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP ?
            REGULARIZER_IDENTIFIER_REGISTRY_SYMBOL_MAP[identifier] :
            identifier;
        const config = { className, config: {} };
        return deserializeRegularizer(config);
    }
    else if (identifier instanceof Regularizer) {
        return identifier;
    }
    else {
        return deserializeRegularizer(identifier);
    }
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicmVndWxhcml6ZXJzLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiLi4vLi4vLi4vLi4vLi4vdGZqcy1sYXllcnMvc3JjL3JlZ3VsYXJpemVycy50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQTs7Ozs7Ozs7R0FRRztBQUVILDRDQUE0QztBQUU1QyxPQUFPLEtBQUssR0FBRyxNQUFNLHVCQUF1QixDQUFDO0FBQzdDLE9BQU8sRUFBQyxHQUFHLEVBQUUsR0FBRyxFQUFVLGFBQWEsRUFBRSxHQUFHLEVBQVUsSUFBSSxFQUFFLEtBQUssRUFBQyxNQUFNLHVCQUF1QixDQUFDO0FBQ2hHLE9BQU8sS0FBSyxDQUFDLE1BQU0sd0JBQXdCLENBQUM7QUFDNUMsT0FBTyxFQUFDLHNCQUFzQixFQUFFLG9CQUFvQixFQUFDLE1BQU0sdUJBQXVCLENBQUM7QUFFbkYsU0FBUyxnQkFBZ0IsQ0FBQyxJQUE0QjtJQUNwRCxJQUFJLElBQUksSUFBSSxJQUFJLElBQUksT0FBTyxJQUFJLEtBQUssUUFBUSxFQUFFO1FBQzVDLE1BQU0sSUFBSSxLQUFLLENBQ1gsa0VBQWtFO1lBQ2xFLHlCQUF5QixJQUFJLEVBQUUsQ0FBQyxDQUFDO0tBQ3RDO0FBQ0gsQ0FBQztBQUVEOztHQUVHO0FBQ0gsTUFBTSxPQUFnQixXQUFZLFNBQVEsYUFBYSxDQUFDLFlBQVk7Q0FFbkU7QUFtQkQsTUFBTSxPQUFPLElBQUssU0FBUSxXQUFXO0lBUW5DLFlBQVksSUFBZTtRQUN6QixLQUFLLEVBQUUsQ0FBQztRQUVSLGdCQUFnQixDQUFDLElBQUksQ0FBQyxDQUFDO1FBRXZCLElBQUksQ0FBQyxFQUFFLEdBQUcsSUFBSSxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDO1FBQzNELElBQUksQ0FBQyxFQUFFLEdBQUcsSUFBSSxJQUFJLElBQUksSUFBSSxJQUFJLENBQUMsRUFBRSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLENBQUMsQ0FBQyxJQUFJLENBQUMsRUFBRSxDQUFDO1FBQzNELElBQUksQ0FBQyxLQUFLLEdBQUcsSUFBSSxDQUFDLEVBQUUsS0FBSyxDQUFDLENBQUM7UUFDM0IsSUFBSSxDQUFDLEtBQUssR0FBRyxJQUFJLENBQUMsRUFBRSxLQUFLLENBQUMsQ0FBQztJQUM3QixDQUFDO0lBRUQ7OztPQUdHO0lBQ0gsS0FBSyxDQUFDLENBQVM7UUFDYixPQUFPLElBQUksQ0FBQyxHQUFHLEVBQUU7WUFDZixJQUFJLGNBQWMsR0FBVyxLQUFLLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO1lBQ3hDLElBQUksSUFBSSxDQUFDLEtBQUssRUFBRTtnQkFDZCxjQUFjLEdBQUcsR0FBRyxDQUFDLGNBQWMsRUFBRSxHQUFHLENBQUMsR0FBRyxDQUFDLEdBQUcsQ0FBQyxJQUFJLENBQUMsRUFBRSxFQUFFLEdBQUcsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQzthQUNyRTtZQUNELElBQUksSUFBSSxDQUFDLEtBQUssRUFBRTtnQkFDZCxjQUFjO29CQUNWLEdBQUcsQ0FBQyxjQUFjLEVBQUUsR0FBRyxDQUFDLEdBQUcsQ0FBQyxHQUFHLENBQUMsSUFBSSxDQUFDLEVBQUUsRUFBRSxDQUFDLENBQUMsTUFBTSxDQUFDLENBQUMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxDQUFDO2FBQzdEO1lBQ0QsT0FBTyxHQUFHLENBQUMsT0FBTyxDQUFDLGNBQWMsRUFBRSxFQUFFLENBQUMsQ0FBQztRQUN6QyxDQUFDLENBQUMsQ0FBQztJQUNMLENBQUM7SUFFRCxTQUFTO1FBQ1AsT0FBTyxFQUFDLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxFQUFFLElBQUksRUFBRSxJQUFJLENBQUMsRUFBRSxFQUFDLENBQUM7SUFDeEMsQ0FBQztJQUVELGtCQUFrQjtJQUNsQixNQUFNLENBQVUsVUFBVSxDQUN0QixHQUE2QyxFQUM3QyxNQUFnQztRQUNsQyxPQUFPLElBQUksR0FBRyxDQUFDLEVBQUMsRUFBRSxFQUFFLE1BQU0sQ0FBQyxJQUFJLENBQVcsRUFBRSxFQUFFLEVBQUUsTUFBTSxDQUFDLElBQUksQ0FBVyxFQUFDLENBQUMsQ0FBQztJQUMzRSxDQUFDOztBQTdDRCxrQkFBa0I7QUFDWCxjQUFTLEdBQUcsTUFBTSxDQUFDO0FBOEM1QixhQUFhLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQyxDQUFDO0FBRWxDLE1BQU0sVUFBVSxFQUFFLENBQUMsSUFBYTtJQUM5QixnQkFBZ0IsQ0FBQyxJQUFJLENBQUMsQ0FBQztJQUN2QixPQUFPLElBQUksSUFBSSxDQUFDLEVBQUMsRUFBRSxFQUFFLElBQUksSUFBSSxJQUFJLENBQUMsQ0FBQyxDQUFDLElBQUksQ0FBQyxFQUFFLENBQUMsQ0FBQyxDQUFDLElBQUksRUFBRSxFQUFFLEVBQUUsQ0FBQyxFQUFDLENBQUMsQ0FBQztBQUM5RCxDQUFDO0FBRUQsTUFBTSxVQUFVLEVBQUUsQ0FBQyxJQUFZO0lBQzdCLGdCQUFnQixDQUFDLElBQUksQ0FBQyxDQUFDO0lBQ3ZCLE9BQU8sSUFBSSxJQUFJLENBQUMsRUFBQyxFQUFFLEVBQUUsSUFBSSxJQUFJLElBQUksQ0FBQyxDQUFDLENBQUMsSUFBSSxDQUFDLEVBQUUsQ0FBQyxDQUFDLENBQUMsSUFBSSxFQUFFLEVBQUUsRUFBRSxDQUFDLEVBQUMsQ0FBQyxDQUFDO0FBQzlELENBQUM7QUFLRCwrRUFBK0U7QUFDL0UsTUFBTSxDQUFDLE1BQU0sMENBQTBDLEdBQ0Q7SUFDaEQsTUFBTSxFQUFFLE1BQU07Q0FDZixDQUFDO0FBRU4sTUFBTSxVQUFVLG9CQUFvQixDQUFDLFVBQXVCO0lBRTFELE9BQU8sb0JBQW9CLENBQUMsVUFBVSxDQUFDLENBQUM7QUFDMUMsQ0FBQztBQUVELE1BQU0sVUFBVSxzQkFBc0IsQ0FDbEMsTUFBZ0MsRUFDaEMsZ0JBQTBDLEVBQUU7SUFDOUMsT0FBTyxzQkFBc0IsQ0FDekIsTUFBTSxFQUFFLGFBQWEsQ0FBQyxnQkFBZ0IsQ0FBQyxNQUFNLEVBQUUsQ0FBQyxZQUFZLEVBQzVELGFBQWEsRUFBRSxhQUFhLENBQUMsQ0FBQztBQUNwQyxDQUFDO0FBRUQsTUFBTSxVQUFVLGNBQWMsQ0FBQyxVQUVXO0lBQ3hDLElBQUksVUFBVSxJQUFJLElBQUksRUFBRTtRQUN0QixPQUFPLElBQUksQ0FBQztLQUNiO0lBQ0QsSUFBSSxPQUFPLFVBQVUsS0FBSyxRQUFRLEVBQUU7UUFDbEMsTUFBTSxTQUFTLEdBQUcsVUFBVSxJQUFJLDBDQUEwQyxDQUFDLENBQUM7WUFDeEUsMENBQTBDLENBQUMsVUFBVSxDQUFDLENBQUMsQ0FBQztZQUN4RCxVQUFVLENBQUM7UUFDZixNQUFNLE1BQU0sR0FBRyxFQUFDLFNBQVMsRUFBRSxNQUFNLEVBQUUsRUFBRSxFQUFDLENBQUM7UUFDdkMsT0FBTyxzQkFBc0IsQ0FBQyxNQUFNLENBQUMsQ0FBQztLQUN2QztTQUFNLElBQUksVUFBVSxZQUFZLFdBQVcsRUFBRTtRQUM1QyxPQUFPLFVBQVUsQ0FBQztLQUNuQjtTQUFNO1FBQ0wsT0FBTyxzQkFBc0IsQ0FBQyxVQUFVLENBQUMsQ0FBQztLQUMzQztBQUNILENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIvKipcbiAqIEBsaWNlbnNlXG4gKiBDb3B5cmlnaHQgMjAxOCBHb29nbGUgTExDXG4gKlxuICogVXNlIG9mIHRoaXMgc291cmNlIGNvZGUgaXMgZ292ZXJuZWQgYnkgYW4gTUlULXN0eWxlXG4gKiBsaWNlbnNlIHRoYXQgY2FuIGJlIGZvdW5kIGluIHRoZSBMSUNFTlNFIGZpbGUgb3IgYXRcbiAqIGh0dHBzOi8vb3BlbnNvdXJjZS5vcmcvbGljZW5zZXMvTUlULlxuICogPT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT09PT1cbiAqL1xuXG4vKiBvcmlnaW5hbCBzb3VyY2U6IGtlcmFzL3JlZ3VsYXJpemVycy5weSAqL1xuXG5pbXBvcnQgKiBhcyB0ZmMgZnJvbSAnQHRlbnNvcmZsb3cvdGZqcy1jb3JlJztcbmltcG9ydCB7YWJzLCBhZGQsIFNjYWxhciwgc2VyaWFsaXphdGlvbiwgc3VtLCBUZW5zb3IsIHRpZHksIHplcm9zfSBmcm9tICdAdGVuc29yZmxvdy90ZmpzLWNvcmUnO1xuaW1wb3J0ICogYXMgSyBmcm9tICcuL2JhY2tlbmQvdGZqc19iYWNrZW5kJztcbmltcG9ydCB7ZGVzZXJpYWxpemVLZXJhc09iamVjdCwgc2VyaWFsaXplS2VyYXNPYmplY3R9IGZyb20gJy4vdXRpbHMvZ2VuZXJpY191dGlscyc7XG5cbmZ1bmN0aW9uIGFzc2VydE9iamVjdEFyZ3MoYXJnczogTDFBcmdzfEwyQXJnc3xMMUwyQXJncyk6IHZvaWQge1xuICBpZiAoYXJncyAhPSBudWxsICYmIHR5cGVvZiBhcmdzICE9PSAnb2JqZWN0Jykge1xuICAgIHRocm93IG5ldyBFcnJvcihcbiAgICAgICAgYEFyZ3VtZW50IHRvIEwxTDIgcmVndWxhcml6ZXIncyBjb25zdHJ1Y3RvciBpcyBleHBlY3RlZCB0byBiZSBhbiBgICtcbiAgICAgICAgYG9iamVjdCwgYnV0IHJlY2VpdmVkOiAke2FyZ3N9YCk7XG4gIH1cbn1cblxuLyoqXG4gKiBSZWd1bGFyaXplciBiYXNlIGNsYXNzLlxuICovXG5leHBvcnQgYWJzdHJhY3QgY2xhc3MgUmVndWxhcml6ZXIgZXh0ZW5kcyBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZSB7XG4gIGFic3RyYWN0IGFwcGx5KHg6IFRlbnNvcik6IFNjYWxhcjtcbn1cblxuZXhwb3J0IGludGVyZmFjZSBMMUwyQXJncyB7XG4gIC8qKiBMMSByZWd1bGFyaXphdGlvbiByYXRlLiBEZWZhdWx0cyB0byAwLjAxLiAqL1xuICBsMT86IG51bWJlcjtcbiAgLyoqIEwyIHJlZ3VsYXJpemF0aW9uIHJhdGUuIERlZmF1bHRzIHRvIDAuMDEuICovXG4gIGwyPzogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIEwxQXJncyB7XG4gIC8qKiBMMSByZWd1bGFyaXphdGlvbiByYXRlLiBEZWZhdWx0cyB0byAwLjAxLiAqL1xuICBsMTogbnVtYmVyO1xufVxuXG5leHBvcnQgaW50ZXJmYWNlIEwyQXJncyB7XG4gIC8qKiBMMiByZWd1bGFyaXphdGlvbiByYXRlLiBEZWZhdWx0cyB0byAwLjAxLiAqL1xuICBsMjogbnVtYmVyO1xufVxuXG5leHBvcnQgY2xhc3MgTDFMMiBleHRlbmRzIFJlZ3VsYXJpemVyIHtcbiAgLyoqIEBub2NvbGxhcHNlICovXG4gIHN0YXRpYyBjbGFzc05hbWUgPSAnTDFMMic7XG5cbiAgcHJpdmF0ZSByZWFkb25seSBsMTogbnVtYmVyO1xuICBwcml2YXRlIHJlYWRvbmx5IGwyOiBudW1iZXI7XG4gIHByaXZhdGUgcmVhZG9ubHkgaGFzTDE6IGJvb2xlYW47XG4gIHByaXZhdGUgcmVhZG9ubHkgaGFzTDI6IGJvb2xlYW47XG4gIGNvbnN0cnVjdG9yKGFyZ3M/OiBMMUwyQXJncykge1xuICAgIHN1cGVyKCk7XG5cbiAgICBhc3NlcnRPYmplY3RBcmdzKGFyZ3MpO1xuXG4gICAgdGhpcy5sMSA9IGFyZ3MgPT0gbnVsbCB8fCBhcmdzLmwxID09IG51bGwgPyAwLjAxIDogYXJncy5sMTtcbiAgICB0aGlzLmwyID0gYXJncyA9PSBudWxsIHx8IGFyZ3MubDIgPT0gbnVsbCA/IDAuMDEgOiBhcmdzLmwyO1xuICAgIHRoaXMuaGFzTDEgPSB0aGlzLmwxICE9PSAwO1xuICAgIHRoaXMuaGFzTDIgPSB0aGlzLmwyICE9PSAwO1xuICB9XG5cbiAgLyoqXG4gICAqIFBvcnRpbmcgbm90ZTogUmVuYW1lZCBmcm9tIF9fY2FsbF9fLlxuICAgKiBAcGFyYW0geCBWYXJpYWJsZSBvZiB3aGljaCB0byBjYWxjdWxhdGUgdGhlIHJlZ3VsYXJpemF0aW9uIHNjb3JlLlxuICAgKi9cbiAgYXBwbHkoeDogVGVuc29yKTogU2NhbGFyIHtcbiAgICByZXR1cm4gdGlkeSgoKSA9PiB7XG4gICAgICBsZXQgcmVndWxhcml6YXRpb246IFRlbnNvciA9IHplcm9zKFsxXSk7XG4gICAgICBpZiAodGhpcy5oYXNMMSkge1xuICAgICAgICByZWd1bGFyaXphdGlvbiA9IGFkZChyZWd1bGFyaXphdGlvbiwgc3VtKHRmYy5tdWwodGhpcy5sMSwgYWJzKHgpKSkpO1xuICAgICAgfVxuICAgICAgaWYgKHRoaXMuaGFzTDIpIHtcbiAgICAgICAgcmVndWxhcml6YXRpb24gPVxuICAgICAgICAgICAgYWRkKHJlZ3VsYXJpemF0aW9uLCBzdW0odGZjLm11bCh0aGlzLmwyLCBLLnNxdWFyZSh4KSkpKTtcbiAgICAgIH1cbiAgICAgIHJldHVybiB0ZmMucmVzaGFwZShyZWd1bGFyaXphdGlvbiwgW10pO1xuICAgIH0pO1xuICB9XG5cbiAgZ2V0Q29uZmlnKCk6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCB7XG4gICAgcmV0dXJuIHsnbDEnOiB0aGlzLmwxLCAnbDInOiB0aGlzLmwyfTtcbiAgfVxuXG4gIC8qKiBAbm9jb2xsYXBzZSAqL1xuICBzdGF0aWMgb3ZlcnJpZGUgZnJvbUNvbmZpZzxUIGV4dGVuZHMgc2VyaWFsaXphdGlvbi5TZXJpYWxpemFibGU+KFxuICAgICAgY2xzOiBzZXJpYWxpemF0aW9uLlNlcmlhbGl6YWJsZUNvbnN0cnVjdG9yPFQ+LFxuICAgICAgY29uZmlnOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QpOiBUIHtcbiAgICByZXR1cm4gbmV3IGNscyh7bDE6IGNvbmZpZ1snbDEnXSBhcyBudW1iZXIsIGwyOiBjb25maWdbJ2wyJ10gYXMgbnVtYmVyfSk7XG4gIH1cbn1cbnNlcmlhbGl6YXRpb24ucmVnaXN0ZXJDbGFzcyhMMUwyKTtcblxuZXhwb3J0IGZ1bmN0aW9uIGwxKGFyZ3M/OiBMMUFyZ3MpIHtcbiAgYXNzZXJ0T2JqZWN0QXJncyhhcmdzKTtcbiAgcmV0dXJuIG5ldyBMMUwyKHtsMTogYXJncyAhPSBudWxsID8gYXJncy5sMSA6IG51bGwsIGwyOiAwfSk7XG59XG5cbmV4cG9ydCBmdW5jdGlvbiBsMihhcmdzOiBMMkFyZ3MpIHtcbiAgYXNzZXJ0T2JqZWN0QXJncyhhcmdzKTtcbiAgcmV0dXJuIG5ldyBMMUwyKHtsMjogYXJncyAhPSBudWxsID8gYXJncy5sMiA6IG51bGwsIGwxOiAwfSk7XG59XG5cbi8qKiBAZG9jaW5saW5lICovXG5leHBvcnQgdHlwZSBSZWd1bGFyaXplcklkZW50aWZpZXIgPSAnbDFsMid8c3RyaW5nO1xuXG4vLyBNYXBzIHRoZSBKYXZhU2NyaXB0LWxpa2UgaWRlbnRpZmllciBrZXlzIHRvIHRoZSBjb3JyZXNwb25kaW5nIGtlcmFzIHN5bWJvbHMuXG5leHBvcnQgY29uc3QgUkVHVUxBUklaRVJfSURFTlRJRklFUl9SRUdJU1RSWV9TWU1CT0xfTUFQOlxuICAgIHtbaWRlbnRpZmllciBpbiBSZWd1bGFyaXplcklkZW50aWZpZXJdOiBzdHJpbmd9ID0ge1xuICAgICAgJ2wxbDInOiAnTDFMMidcbiAgICB9O1xuXG5leHBvcnQgZnVuY3Rpb24gc2VyaWFsaXplUmVndWxhcml6ZXIoY29uc3RyYWludDogUmVndWxhcml6ZXIpOlxuICAgIHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdFZhbHVlIHtcbiAgcmV0dXJuIHNlcmlhbGl6ZUtlcmFzT2JqZWN0KGNvbnN0cmFpbnQpO1xufVxuXG5leHBvcnQgZnVuY3Rpb24gZGVzZXJpYWxpemVSZWd1bGFyaXplcihcbiAgICBjb25maWc6IHNlcmlhbGl6YXRpb24uQ29uZmlnRGljdCxcbiAgICBjdXN0b21PYmplY3RzOiBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3QgPSB7fSk6IFJlZ3VsYXJpemVyIHtcbiAgcmV0dXJuIGRlc2VyaWFsaXplS2VyYXNPYmplY3QoXG4gICAgICBjb25maWcsIHNlcmlhbGl6YXRpb24uU2VyaWFsaXphdGlvbk1hcC5nZXRNYXAoKS5jbGFzc05hbWVNYXAsXG4gICAgICBjdXN0b21PYmplY3RzLCAncmVndWxhcml6ZXInKTtcbn1cblxuZXhwb3J0IGZ1bmN0aW9uIGdldFJlZ3VsYXJpemVyKGlkZW50aWZpZXI6IFJlZ3VsYXJpemVySWRlbnRpZmllcnxcbiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBzZXJpYWxpemF0aW9uLkNvbmZpZ0RpY3R8XG4gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgUmVndWxhcml6ZXIpOiBSZWd1bGFyaXplciB7XG4gIGlmIChpZGVudGlmaWVyID09IG51bGwpIHtcbiAgICByZXR1cm4gbnVsbDtcbiAgfVxuICBpZiAodHlwZW9mIGlkZW50aWZpZXIgPT09ICdzdHJpbmcnKSB7XG4gICAgY29uc3QgY2xhc3NOYW1lID0gaWRlbnRpZmllciBpbiBSRUdVTEFSSVpFUl9JREVOVElGSUVSX1JFR0lTVFJZX1NZTUJPTF9NQVAgP1xuICAgICAgICBSRUdVTEFSSVpFUl9JREVOVElGSUVSX1JFR0lTVFJZX1NZTUJPTF9NQVBbaWRlbnRpZmllcl0gOlxuICAgICAgICBpZGVudGlmaWVyO1xuICAgIGNvbnN0IGNvbmZpZyA9IHtjbGFzc05hbWUsIGNvbmZpZzoge319O1xuICAgIHJldHVybiBkZXNlcmlhbGl6ZVJlZ3VsYXJpemVyKGNvbmZpZyk7XG4gIH0gZWxzZSBpZiAoaWRlbnRpZmllciBpbnN0YW5jZW9mIFJlZ3VsYXJpemVyKSB7XG4gICAgcmV0dXJuIGlkZW50aWZpZXI7XG4gIH0gZWxzZSB7XG4gICAgcmV0dXJuIGRlc2VyaWFsaXplUmVndWxhcml6ZXIoaWRlbnRpZmllcik7XG4gIH1cbn1cbiJdfQ==