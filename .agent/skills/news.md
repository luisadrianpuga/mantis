## Getting news
Fetch RSS headlines. Always use single-line COMMAND: with semicolons.
Never use multi-line subshells. Never use FETCH: for RSS.

Maryland + local + BBC headlines:
COMMAND: curl -sL https://www.marylandmatters.org/feed/ > /tmp/f1.xml; curl -sL https://www.aacounty.org/news/rss.xml > /tmp/f2.xml; curl -sL http://feeds.bbci.co.uk/news/rss.xml > /tmp/f3.xml; cat /tmp/f1.xml /tmp/f2.xml /tmp/f3.xml | awk '/<item>/{i=1} /<\/item>/{i=0} i&&/<title>/{gsub(/.*<title>/,"");gsub(/<\/title>.*/,"");gsub(/<!\[CDATA\[|\]\]>/,"");if(length>2)print}' | nl | head -15

AP News only:
COMMAND: curl -sL https://feeds.apnews.com/rss/apf-topnews | awk '/<item>/{i=1} /<\/item>/{i=0} i&&/<title>/{gsub(/.*<title>/,"");gsub(/<\/title>.*/,"");gsub(/<!\[CDATA\[|\]\]>/,"");if(length>2)print}' | nl | head -6

Return results as a numbered list. If output is empty, tell the user.
