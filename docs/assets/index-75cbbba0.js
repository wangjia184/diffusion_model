(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const l of document.querySelectorAll('link[rel="modulepreload"]'))r(l);new MutationObserver(l=>{for(const s of l)if(s.type==="childList")for(const o of s.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&r(o)}).observe(document,{childList:!0,subtree:!0});function n(l){const s={};return l.integrity&&(s.integrity=l.integrity),l.referrerPolicy&&(s.referrerPolicy=l.referrerPolicy),l.crossOrigin==="use-credentials"?s.credentials="include":l.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function r(l){if(l.ep)return;l.ep=!0;const s=n(l);fetch(l.href,s)}})();function E(){}function I(t,e){for(const n in e)t[n]=e[n];return t}function oe(t){return t()}function ne(){return Object.create(null)}function L(t){t.forEach(oe)}function ie(t){return typeof t=="function"}function ce(t,e){return t!=t?e==e:t!==e||t&&typeof t=="object"||typeof t=="function"}function be(t){return Object.keys(t).length===0}function H(t,e,n,r){if(t){const l=fe(t,e,n,r);return t[0](l)}}function fe(t,e,n,r){return t[1]&&r?I(n.ctx.slice(),t[1](r(e))):n.ctx}function J(t,e,n,r){if(t[2]&&r){const l=t[2](r(n));if(e.dirty===void 0)return l;if(typeof l=="object"){const s=[],o=Math.max(e.dirty.length,l.length);for(let f=0;f<o;f+=1)s[f]=e.dirty[f]|l[f];return s}return e.dirty|l}return e.dirty}function Q(t,e,n,r,l,s){if(l){const o=fe(e,n,r,s);t.p(o,l)}}function R(t){if(t.ctx.length>32){const e=[],n=t.ctx.length/32;for(let r=0;r<n;r++)e[r]=-1;return e}return-1}function ye(t){const e={};for(const n in t)n[0]!=="$"&&(e[n]=t[n]);return e}function re(t,e){const n={};e=new Set(e);for(const r in t)!e.has(r)&&r[0]!=="$"&&(n[r]=t[r]);return n}function P(t,e){t.appendChild(e)}function w(t,e,n){t.insertBefore(e,n||null)}function b(t){t.parentNode&&t.parentNode.removeChild(t)}function M(t){return document.createElement(t)}function K(t){return document.createTextNode(t)}function U(){return K(" ")}function ue(){return K("")}function h(t,e,n){n==null?t.removeAttribute(e):t.getAttribute(e)!==n&&t.setAttribute(e,n)}function W(t,e){const n=Object.getOwnPropertyDescriptors(t.__proto__);for(const r in e)e[r]==null?t.removeAttribute(r):r==="style"?t.style.cssText=e[r]:r==="__value"?t.value=t[r]=e[r]:n[r]&&n[r].set?t[r]=e[r]:h(t,r,e[r])}function ke(t){return Array.from(t.childNodes)}function ve(t,e){e=""+e,t.wholeText!==e&&(t.data=e)}function z(t,e,n,r){n===null?t.style.removeProperty(e):t.style.setProperty(e,n,r?"important":"")}let te;function S(t){te=t}const A=[],x=[],F=[],le=[],we=Promise.resolve();let $=!1;function Ne(){$||($=!0,we.then(ae))}function ee(t){F.push(t)}const Z=new Set;let O=0;function ae(){if(O!==0)return;const t=te;do{try{for(;O<A.length;){const e=A[O];O++,S(e),Me(e.$$)}}catch(e){throw A.length=0,O=0,e}for(S(null),A.length=0,O=0;x.length;)x.pop()();for(let e=0;e<F.length;e+=1){const n=F[e];Z.has(n)||(Z.add(n),n())}F.length=0}while(A.length);for(;le.length;)le.pop()();$=!1,Z.clear(),S(t)}function Me(t){if(t.fragment!==null){t.update(),L(t.before_update);const e=t.dirty;t.dirty=[-1],t.fragment&&t.fragment.p(t.ctx,e),t.after_update.forEach(ee)}}const G=new Set;let j;function V(){j={r:0,c:[],p:j}}function X(){j.r||L(j.c),j=j.p}function y(t,e){t&&t.i&&(G.delete(t),t.i(e))}function v(t,e,n,r){if(t&&t.o){if(G.has(t))return;G.add(t),j.c.push(()=>{G.delete(t),r&&(n&&t.d(1),r())}),t.o(e)}else r&&r()}function de(t,e){const n={},r={},l={$$scope:1};let s=t.length;for(;s--;){const o=t[s],f=e[s];if(f){for(const i in o)i in f||(r[i]=1);for(const i in f)l[i]||(n[i]=f[i],l[i]=1);t[s]=f}else for(const i in o)l[i]=1}for(const o in r)o in n||(n[o]=void 0);return n}function Ce(t){t&&t.c()}function _e(t,e,n,r){const{fragment:l,after_update:s}=t.$$;l&&l.m(e,n),r||ee(()=>{const o=t.$$.on_mount.map(oe).filter(ie);t.$$.on_destroy?t.$$.on_destroy.push(...o):L(o),t.$$.on_mount=[]}),s.forEach(ee)}function me(t,e){const n=t.$$;n.fragment!==null&&(L(n.on_destroy),n.fragment&&n.fragment.d(e),n.on_destroy=n.fragment=null,n.ctx=[])}function Pe(t,e){t.$$.dirty[0]===-1&&(A.push(t),Ne(),t.$$.dirty.fill(0)),t.$$.dirty[e/31|0]|=1<<e%31}function pe(t,e,n,r,l,s,o,f=[-1]){const i=te;S(t);const c=t.$$={fragment:null,ctx:[],props:s,update:E,not_equal:l,bound:ne(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(e.context||(i?i.$$.context:[])),callbacks:ne(),dirty:f,skip_bound:!1,root:e.target||i.$$.root};o&&o(c.root);let a=!1;if(c.ctx=n?n(t,e.props||{},(u,p,...k)=>{const C=k.length?k[0]:p;return c.ctx&&l(c.ctx[u],c.ctx[u]=C)&&(!c.skip_bound&&c.bound[u]&&c.bound[u](C),a&&Pe(t,u)),p}):[],c.update(),a=!0,L(c.before_update),c.fragment=r?r(c.ctx):!1,e.target){if(e.hydrate){const u=ke(e.target);c.fragment&&c.fragment.l(u),u.forEach(b)}else c.fragment&&c.fragment.c();e.intro&&y(t.$$.fragment),_e(t,e.target,e.anchor,e.customElement),ae()}S(i)}class ge{$destroy(){me(this,1),this.$destroy=E}$on(e,n){if(!ie(n))return E;const r=this.$$.callbacks[e]||(this.$$.callbacks[e]=[]);return r.push(n),()=>{const l=r.indexOf(n);l!==-1&&r.splice(l,1)}}$set(e){this.$$set&&!be(e)&&(this.$$.skip_bound=!0,this.$$set(e),this.$$.skip_bound=!1)}}function he(t){let e="";if(typeof t=="string"||typeof t=="number")e+=t;else if(typeof t=="object")if(Array.isArray(t))e=t.map(he).filter(Boolean).join(" ");else for(let n in t)t[n]&&(e&&(e+=" "),e+=n);return e}function se(...t){return t.map(he).filter(Boolean).join(" ")}function je(t){let e,n,r,l;const s=[Se,Ae],o=[];function f(a,u){return a[1]?0:1}n=f(t),r=o[n]=s[n](t);let i=[t[7],{class:t[6]}],c={};for(let a=0;a<i.length;a+=1)c=I(c,i[a]);return{c(){e=M("div"),r.c(),W(e,c)},m(a,u){w(a,e,u),o[n].m(e,null),l=!0},p(a,u){let p=n;n=f(a),n===p?o[n].p(a,u):(V(),v(o[p],1,1,()=>{o[p]=null}),X(),r=o[n],r?r.p(a,u):(r=o[n]=s[n](a),r.c()),y(r,1),r.m(e,null)),W(e,c=de(i,[u&128&&a[7],(!l||u&64)&&{class:a[6]}]))},i(a){l||(y(r),l=!0)},o(a){v(r),l=!1},d(a){a&&b(e),o[n].d()}}}function Oe(t){let e,n,r,l;const s=[Ie,Ee],o=[];function f(i,c){return i[1]?0:1}return e=f(t),n=o[e]=s[e](t),{c(){n.c(),r=ue()},m(i,c){o[e].m(i,c),w(i,r,c),l=!0},p(i,c){let a=e;e=f(i),e===a?o[e].p(i,c):(V(),v(o[a],1,1,()=>{o[a]=null}),X(),n=o[e],n?n.p(i,c):(n=o[e]=s[e](i),n.c()),y(n,1),n.m(r.parentNode,r))},i(i){l||(y(n),l=!0)},o(i){v(n),l=!1},d(i){o[e].d(i),i&&b(r)}}}function Ae(t){let e,n;const r=t[14].default,l=H(r,t,t[13],null);return{c(){e=M("div"),l&&l.c(),h(e,"class",t[5]),z(e,"width",t[4]+"%"),h(e,"role","progressbar"),h(e,"aria-valuenow",t[2]),h(e,"aria-valuemin","0"),h(e,"aria-valuemax",t[3])},m(s,o){w(s,e,o),l&&l.m(e,null),n=!0},p(s,o){l&&l.p&&(!n||o&8192)&&Q(l,r,s,s[13],n?J(r,s[13],o,null):R(s[13]),null),(!n||o&32)&&h(e,"class",s[5]),(!n||o&16)&&z(e,"width",s[4]+"%"),(!n||o&4)&&h(e,"aria-valuenow",s[2]),(!n||o&8)&&h(e,"aria-valuemax",s[3])},i(s){n||(y(l,s),n=!0)},o(s){v(l,s),n=!1},d(s){s&&b(e),l&&l.d(s)}}}function Se(t){let e;const n=t[14].default,r=H(n,t,t[13],null);return{c(){r&&r.c()},m(l,s){r&&r.m(l,s),e=!0},p(l,s){r&&r.p&&(!e||s&8192)&&Q(r,n,l,l[13],e?J(n,l[13],s,null):R(l[13]),null)},i(l){e||(y(r,l),e=!0)},o(l){v(r,l),e=!1},d(l){r&&r.d(l)}}}function Ee(t){let e,n,r;const l=t[14].default,s=H(l,t,t[13],null);let o=[t[7],{class:t[5]},{style:n="width: "+t[4]+"%"},{role:"progressbar"},{"aria-valuenow":t[2]},{"aria-valuemin":"0"},{"aria-valuemax":t[3]}],f={};for(let i=0;i<o.length;i+=1)f=I(f,o[i]);return{c(){e=M("div"),s&&s.c(),W(e,f)},m(i,c){w(i,e,c),s&&s.m(e,null),r=!0},p(i,c){s&&s.p&&(!r||c&8192)&&Q(s,l,i,i[13],r?J(l,i[13],c,null):R(i[13]),null),W(e,f=de(o,[c&128&&i[7],(!r||c&32)&&{class:i[5]},(!r||c&16&&n!==(n="width: "+i[4]+"%"))&&{style:n},{role:"progressbar"},(!r||c&4)&&{"aria-valuenow":i[2]},{"aria-valuemin":"0"},(!r||c&8)&&{"aria-valuemax":i[3]}]))},i(i){r||(y(s,i),r=!0)},o(i){v(s,i),r=!1},d(i){i&&b(e),s&&s.d(i)}}}function Ie(t){let e;const n=t[14].default,r=H(n,t,t[13],null);return{c(){r&&r.c()},m(l,s){r&&r.m(l,s),e=!0},p(l,s){r&&r.p&&(!e||s&8192)&&Q(r,n,l,l[13],e?J(n,l[13],s,null):R(l[13]),null)},i(l){e||(y(r,l),e=!0)},o(l){v(r,l),e=!1},d(l){r&&r.d(l)}}}function Le(t){let e,n,r,l;const s=[Oe,je],o=[];function f(i,c){return i[0]?0:1}return e=f(t),n=o[e]=s[e](t),{c(){n.c(),r=ue()},m(i,c){o[e].m(i,c),w(i,r,c),l=!0},p(i,[c]){let a=e;e=f(i),e===a?o[e].p(i,c):(V(),v(o[a],1,1,()=>{o[a]=null}),X(),n=o[e],n?n.p(i,c):(n=o[e]=s[e](i),n.c()),y(n,1),n.m(r.parentNode,r))},i(i){l||(y(n),l=!0)},o(i){v(n),l=!1},d(i){o[e].d(i),i&&b(r)}}}function Be(t,e,n){let r,l,s;const o=["class","bar","multi","value","max","animated","striped","color","barClassName"];let f=re(e,o),{$$slots:i={},$$scope:c}=e,{class:a=""}=e,{bar:u=!1}=e,{multi:p=!1}=e,{value:k=0}=e,{max:C=100}=e,{animated:d=!1}=e,{striped:m=!1}=e,{color:g=""}=e,{barClassName:N=""}=e;return t.$$set=_=>{e=I(I({},e),ye(_)),n(7,f=re(e,o)),"class"in _&&n(8,a=_.class),"bar"in _&&n(0,u=_.bar),"multi"in _&&n(1,p=_.multi),"value"in _&&n(2,k=_.value),"max"in _&&n(3,C=_.max),"animated"in _&&n(9,d=_.animated),"striped"in _&&n(10,m=_.striped),"color"in _&&n(11,g=_.color),"barClassName"in _&&n(12,N=_.barClassName),"$$scope"in _&&n(13,c=_.$$scope)},t.$$.update=()=>{t.$$.dirty&256&&n(6,r=se(a,"progress")),t.$$.dirty&7937&&n(5,l=se("progress-bar",u&&a||N,d?"progress-bar-animated":null,g?`bg-${g}`:null,m||d?"progress-bar-striped":null)),t.$$.dirty&12&&n(4,s=parseInt(k,10)/parseInt(C,10)*100)},[u,p,k,C,s,l,r,f,a,d,m,g,N,c,i]}class De extends ge{constructor(e){super(),pe(this,e,Be,Le,ce,{class:8,bar:0,multi:1,value:2,max:3,animated:9,striped:10,color:11,barClassName:12})}}function Te(t){let e,n,r,l,s;return{c(){e=M("div"),n=M("div"),r=M("canvas"),l=U(),s=M("div"),h(r,"class","d-flex"),h(s,"class","d-flex progress_bar svelte-qmb04g"),z(s,"width",t[2]*100+"%"),h(e,"class","d-flex justify-content-center")},m(o,f){w(o,e,f),P(e,n),P(n,r),t[3](r),P(n,l),P(n,s)},p(o,f){f&4&&z(s,"width",o[2]*100+"%")},i:E,o:E,d(o){o&&b(e),t[3](null)}}}function qe(t){let e,n,r,l,s;return l=new De({props:{animated:!0,color:"success",value:t[0],$$slots:{default:[Fe]},$$scope:{ctx:t}}}),{c(){e=M("div"),n=M("div"),n.textContent="Loading ...",r=U(),Ce(l.$$.fragment),h(n,"class","text-center")},m(o,f){w(o,e,f),P(e,n),P(e,r),_e(l,e,null),s=!0},p(o,f){const i={};f&1&&(i.value=o[0]),f&4097&&(i.$$scope={dirty:f,ctx:o}),l.$set(i)},i(o){s||(y(l.$$.fragment,o),s=!0)},o(o){v(l.$$.fragment,o),s=!1},d(o){o&&b(e),me(l)}}}function Fe(t){let e,n;return{c(){e=K(t[0]),n=K("%")},m(r,l){w(r,e,l),w(r,n,l)},p(r,l){l&1&&ve(e,r[0])},d(r){r&&b(e),r&&b(n)}}}function Ge(t){let e,n,r,l,s,o,f;const i=[qe,Te],c=[];function a(u,p){return u[0]<100?0:1}return r=a(t),l=c[r]=i[r](t),{c(){e=M("link"),n=U(),l.c(),s=U(),o=M("main"),h(e,"rel","stylesheet"),h(e,"href","https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css")},m(u,p){P(document.head,e),w(u,n,p),c[r].m(u,p),w(u,s,p),w(u,o,p),f=!0},p(u,[p]){let k=r;r=a(u),r===k?c[r].p(u,p):(V(),v(c[k],1,1,()=>{c[k]=null}),X(),l=c[r],l?l.p(u,p):(l=c[r]=i[r](u),l.c()),y(l,1),l.m(s.parentNode,s))},i(u){f||(y(l),f=!0)},o(u){v(l),f=!1},d(u){b(e),u&&b(n),c[r].d(u),u&&b(s),u&&b(o)}}}function Ke(t,e,n){let r=0,l,s=0,o=0;const f={},i=new Worker("/worker.js");i.onmessage=async d=>{const m=d.data;switch(m.type){case"reply":{const g=f[m.id];g?g.resolve(m.data):console.log("Unable to find the promise.",m);break}case"ready":{n(0,r=100),p();break}case"progress":{n(0,r=Math.floor(m.progress*100));break}case"error":{alert(m.message);break}default:{console.log(m);break}}};const c=()=>{const d=s++,m=new Promise((g,N)=>{f[d]={resolve:g,reject:N}});return{id:d,promise:m}},a=async()=>{const{id:d,promise:m}=c(),g={type:"ddpmStart",id:d,kwargs:{}};return i.postMessage(g),await m},u=async d=>{const{id:m,promise:g}=c(),N={type:"ddpmNext",id:m,kwargs:{key:d}};return i.postMessage(N),await g},p=async()=>{let d=await a();for(k(d.image),n(2,o=d.percent);d.step>0;)d=await u(d.key),k(d.image),n(2,o=d.percent)},k=d=>{const m=d[0].length,g=d[0][0].length;n(1,l.width=g,l),n(1,l.height=m,l);const N=l.getContext("2d"),_=N.createImageData(g,m,{colorSpace:"srgb"}),B=_.data;for(let D=0;D<m;D++)for(let T=0;T<g;T++){const q=(D*g+T)*4,Y=d[0][D][T];B[q]=Math.max(0,Math.min(255,Y[0]*127.5+127.5)),B[q+1]=Math.max(0,Math.min(255,Y[1]*127.5+127.5)),B[q+2]=Math.max(0,Math.min(255,Y[2]*127.5+127.5)),B[q+3]=255}N.putImageData(_,0,0)};function C(d){x[d?"unshift":"push"](()=>{l=d,n(1,l)})}return[r,l,o,C]}class Ue extends ge{constructor(e){super(),pe(this,e,Ke,Ge,ce,{})}}new Ue({target:document.getElementById("app")});